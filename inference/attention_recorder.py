"""Capture attention maps and their gradients for Chefer relevance propagation.

The SigLIP / MatchVision / SoccerMaster scripts wrap each attention module by
hand (see `inference_utils.wrap_siglip_attention_module`). The off-the-shelf
baselines (CLIP, PE-AV, VideoPrism) come from recent `transformers` / `timm`
releases whose attention code differs per model, so this module hooks them
generically instead:

* Hugging Face models with `attn_implementation="eager"` resolve the module
  level function `eager_attention_forward` at call time. `patch_hf_eager`
  swaps it for a numerically identical version that records the softmax
  output.
* timm attention blocks keep an `attn_drop` module that is applied to the
  softmax output on the non-fused path. `hook_timm_attention` disables the
  fused kernel and hooks that module's output.
* timm's `AttentionPoolLatent` has no such module, so `patch_timm_attention_pool`
  replaces its forward with a recording copy.

For every recorded call the gradient hook reduces attention and gradient to
the head-averaged positive contribution of Chefer et al. (paper Eq. 3),

    A_bar = E_h[ ReLU( dlogit/dA  *  A ) ],

so the full gradient tensor never has to be kept.
"""

import types

import torch
import torch.nn.functional as F


class AttentionRecorder:
    def __init__(self):
        self.names = {}
        self.records = []
        self.enabled = False

    def track(self, module, name):
        self.names[id(module)] = name

    def reset(self):
        self.records = []

    def record(self, module, probs):
        """Register `probs` (B, heads, Nq, Nk) for a tracked module."""
        name = self.names.get(id(module))
        if not self.enabled or name is None or not probs.requires_grad:
            return
        entry = {"name": name, "cam": None}
        self.records.append(entry)

        def _hook(grad, probs=probs, entry=entry):
            entry["cam"] = (grad.detach().float() * probs.detach().float()).clamp(min=0).mean(dim=1)

        probs.register_hook(_hook)

    def cams(self, prefix):
        """Head-averaged Chefer maps (B, Nq, Nk) of the tracked modules whose
        name starts with `prefix`, in forward order."""
        out = []
        for entry in self.records:
            if entry["name"].startswith(prefix):
                if entry["cam"] is None:
                    raise RuntimeError(f"no gradient reached attention '{entry['name']}'; was backward() called?")
                out.append(entry["cam"])
        return out


def rollout(cams, num_tokens, batch, device):
    """Chefer relevance accumulation R <- R + A_bar R, starting from identity."""
    relevance = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(batch, 1, 1)
    for cam in cams:
        relevance = relevance + torch.bmm(cam, relevance)
    return relevance


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def patch_hf_eager(modeling_module, recorder):
    """Replace `modeling_module.eager_attention_forward` with a recording copy.

    Mirrors the eager implementations of CLIP, PE-Video and VideoPrism in
    transformers 5.22 (logit soft-capping included). Returns a callable that
    restores the original function.
    """
    original = modeling_module.eager_attention_forward

    def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                                softcap=None, **kwargs):
        if scaling is None:
            scaling = module.head_dim ** -0.5
        n_rep = int(getattr(module, "num_key_value_groups", 1))
        key_states = _repeat_kv(key, n_rep)
        value_states = _repeat_kv(value, n_rep)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        recorder.record(module, probs)
        probs = F.dropout(probs, p=dropout, training=module.training)
        attn_output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
        return attn_output, probs

    modeling_module.eager_attention_forward = eager_attention_forward

    def restore():
        modeling_module.eager_attention_forward = original

    return restore


def hook_timm_attention(attn_module, name, recorder):
    """Record the softmax output of a timm attention block (Attention,
    EvaAttention, AttentionRope) through its `attn_drop` module."""
    attn_module.fused_attn = False
    recorder.track(attn_module.attn_drop, name)
    return attn_module.attn_drop.register_forward_hook(lambda mod, _inp, out: recorder.record(mod, out))


def patch_timm_attention_pool(pool, name, recorder):
    """Replace `timm.layers.AttentionPoolLatent.forward` (timm 1.0.31) with a
    copy that records the latent-query attention."""
    recorder.track(pool, name)

    def forward(self, x, attn_mask=None):
        batch, num_tokens, channels = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
        q_latent = self.latent.expand(batch, -1, -1)
        q = self.q(q_latent).reshape(batch, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).reshape(batch, num_tokens, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        recorder.record(self, attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, self.latent_len, channels)
        x = self.proj_drop(self.proj(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm(x))
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x

    pool.forward = types.MethodType(forward, pool)
