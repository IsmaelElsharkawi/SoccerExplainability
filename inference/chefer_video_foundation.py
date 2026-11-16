"""Joint-ST attribution for the released PE-AV and VideoPrism video branches."""

from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from chefer_off_the_shelf import _resolve_target


class AttentionCapture:
    """Retain one attention module's latest probabilities and gradients."""

    def __init__(self, module, name):
        self.name = name
        self.attention = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                f"{self.name} did not return (output, attention_weights)"
            )
        attention = output[1]
        if attention is None:
            raise RuntimeError(
                f"{self.name} returned no attention weights; eager attention must be enabled"
            )
        self.attention = attention
        if attention.requires_grad:
            attention.retain_grad()

    def cam(self):
        if self.attention is None:
            raise RuntimeError(f"No attention was captured for {self.name}")
        if self.attention.grad is None:
            raise RuntimeError(f"No attention gradient was captured for {self.name}")
        return (
            (self.attention.grad.float() * self.attention.float())
            .clamp(min=0)
            .mean(dim=1)
        )

    def close(self):
        self.handle.remove()


def _store_attention(module, attention):
    module._chefer_attention = attention
    if attention.requires_grad:
        attention.retain_grad()


def _module_cam(module, name):
    attention = getattr(module, "_chefer_attention", None)
    if attention is None:
        raise RuntimeError(f"No attention was captured for {name}")
    if attention.grad is None:
        raise RuntimeError(f"No attention gradient was captured for {name}")
    return (attention.grad.float() * attention.float()).clamp(min=0).mean(dim=1)


def wrap_timm_eva_attention(module):
    """Expose the eager attention used by timm's PE-Core EVA blocks."""
    if hasattr(module, "_chefer_wrapped"):
        return module
    try:
        from timm.layers import (
            apply_rot_embed_cat,
            maybe_add_mask,
            resolve_self_attn_mask,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The pinned PE-AV environment requires timm with EVA/PE-Core support"
        ) from exc

    def wrapped_forward(self, x, rope=None, attn_mask=None, is_causal=False):
        (batch_size, num_tokens, channels) = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        if self.qkv is not None:
            if not hasattr(self, "q_bias"):
                qkv = self.qkv(x)
            elif self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv = qkv + qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(
                batch_size, num_tokens, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            (query, key, value) = qkv.unbind(0)
        else:
            query = (
                self.q_proj(x)
                .reshape(batch_size, num_tokens, self.num_heads, -1)
                .transpose(1, 2)
            )
            key = (
                self.k_proj(x)
                .reshape(batch_size, num_tokens, self.num_heads, -1)
                .transpose(1, 2)
            )
            value = (
                self.v_proj(x)
                .reshape(batch_size, num_tokens, self.num_heads, -1)
                .transpose(1, 2)
            )
        (query, key) = (self.q_norm(query), self.k_norm(key))
        if rope is not None:
            num_prefix = self.num_prefix_tokens
            rotate_half = getattr(self, "rotate_half", False)
            query = torch.cat(
                [
                    query[:, :, :num_prefix, :],
                    apply_rot_embed_cat(
                        query[:, :, num_prefix:, :], rope, half=rotate_half
                    ),
                ],
                dim=2,
            ).type_as(value)
            key = torch.cat(
                [
                    key[:, :, :num_prefix, :],
                    apply_rot_embed_cat(
                        key[:, :, num_prefix:, :], rope, half=rotate_half
                    ),
                ],
                dim=2,
            ).type_as(value)
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)
        attention_bias = resolve_self_attn_mask(
            num_tokens, attention, attn_mask, is_causal=is_causal
        )
        attention = maybe_add_mask(attention, attention_bias)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        _store_attention(self, attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(
            batch_size, num_tokens, getattr(self, "attn_dim", channels)
        )
        output = self.norm(output)
        if gate is not None:
            output = output * gate
        output = self.proj(output)
        return self.proj_drop(output)

    module.forward = MethodType(wrapped_forward, module)
    module._chefer_wrapped = True
    return module


def wrap_timm_attention_pool(module):
    """Expose timm AttentionPoolLatent's learned-query cross-attention."""
    if hasattr(module, "_chefer_wrapped"):
        return module
    try:
        from timm.layers import maybe_add_mask
    except ImportError as exc:
        raise RuntimeError(
            "The pinned PE-AV environment requires timm AttentionPoolLatent"
        ) from exc

    def wrapped_forward(self, x, attn_mask=None):
        (batch_size, num_tokens, channels) = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
        query_latent = self.latent.expand(batch_size, -1, -1)
        query = (
            self.q(query_latent)
            .reshape(batch_size, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_value = (
            self.kv(x)
            .reshape(batch_size, num_tokens, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        (key, value) = key_value.unbind(0)
        (query, key) = (self.q_norm(query), self.k_norm(key))
        attention = query * self.scale @ key.transpose(-2, -1)
        attention = maybe_add_mask(attention, attn_mask)
        attention = attention.softmax(dim=-1)
        _store_attention(self, attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, self.latent_len, channels)
        output = self.proj(output)
        output = self.proj_drop(output)
        if self.mlp is not None:
            output = output + self.mlp(self.norm(output))
        if self.pool == "token":
            output = output[:, 0]
        elif self.pool == "avg":
            output = output.mean(1)
        return output

    module.forward = MethodType(wrapped_forward, module)
    module._chefer_wrapped = True
    return module


def _propagate_rows(seed, cams):
    """Compute ``seed @ Π_l(I + C_l)`` without materializing full rollout.

    Executable Chefer rollout updates ``R <- R + C_l @ R`` in forward layer
    order, so output rows are multiplied by the CAM matrices in reverse order.
    ``seed`` is ``[B,Q,N]`` and each CAM is ``[B,N,N]``.
    """
    relevance = seed.float()
    for cam in reversed(cams):
        if cam.shape[0] != relevance.shape[0]:
            raise RuntimeError(
                f"CAM batch {cam.shape[0]} does not match relevance batch {relevance.shape[0]}"
            )
        relevance = relevance + torch.bmm(relevance, cam.float())
    return relevance


def _reshape_patch_maps(patch_relevance, grid_size):
    (num_frames, num_patches) = patch_relevance.shape
    if num_patches != grid_size * grid_size:
        raise RuntimeError(
            f"Cannot reshape {num_patches} patches to {grid_size}x{grid_size}"
        )
    return patch_relevance.reshape(num_frames, grid_size, grid_size)


def generate_peav_heatmaps(model, pixel_values_videos, target_label=None):
    """Return raw Joint-ST heatmaps, frame relevance, logits, target, and details."""
    model.eval()
    if pixel_values_videos.dim() != 5 or pixel_values_videos.shape[0] != 1:
        raise ValueError("PE-AV Chefer expects batch-size-one [B,T,C,H,W] input")
    spatial_encoder = model.spatial_encoder
    spatial_modules = []
    for layer_idx, block in enumerate(spatial_encoder.blocks):
        spatial_modules.append(wrap_timm_eva_attention(block.attn))
    pool_module = wrap_timm_attention_pool(spatial_encoder.attn_pool)
    temporal_captures = []
    temporal_captures = [
        AttentionCapture(layer.self_attn, f"PE-AV temporal layer {layer_idx}")
        for (layer_idx, layer) in enumerate(model.temporal_encoder.layers)
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(pixel_values_videos)
        target_label = _resolve_target(logits, target_label)
        logits[0, target_label].backward()
        spatial_cams = [
            _module_cam(module, f"PE-AV spatial layer {layer_idx}")
            for (layer_idx, module) in enumerate(spatial_modules)
        ]
        pool_cam = _module_cam(pool_module, "PE-AV spatial attention pool")
        pool_token_relevance = _propagate_rows(pool_cam, spatial_cams).squeeze(1)
        num_frames = pixel_values_videos.shape[1]
        if pool_token_relevance.shape[0] != num_frames:
            raise RuntimeError(
                f"PE-AV spatial leading dimension is {pool_token_relevance.shape[0]}, expected {num_frames}"
            )
        num_prefix_tokens = int(spatial_encoder.num_prefix_tokens)
        patch_relevance = pool_token_relevance[:, num_prefix_tokens:]
        grid_size = spatial_encoder.patch_embed.grid_size[0]
        spatial_raw = _reshape_patch_maps(patch_relevance, grid_size)
        spatial_raw_numpy = spatial_raw.detach().cpu().numpy().astype(np.float32)
        heatmaps = spatial_raw_numpy
        raw_temporal_weights = torch.ones(
            num_frames, dtype=torch.float32, device=pixel_values_videos.device
        )
        temporal_weights = raw_temporal_weights
        temporal_cams = [capture.cam() for capture in temporal_captures]
        num_temporal_tokens = num_frames + 1
        cls_seed = torch.zeros(
            1, 1, num_temporal_tokens, device=pixel_values_videos.device
        )
        cls_seed[:, :, 0] = 1.0
        temporal_relevance = (
            _propagate_rows(cls_seed, temporal_cams).squeeze(0).squeeze(0)
        )
        raw_temporal_weights = temporal_relevance[1:]
        temporal_weights = raw_temporal_weights
        heatmaps = (
            spatial_raw_numpy
            * raw_temporal_weights.detach().cpu().numpy()[:, None, None]
        )
        details = {
            "spatial_raw": spatial_raw_numpy,
            "pool_token_relevance": pool_token_relevance.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "raw_temporal_weights": raw_temporal_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "temporal_weights": temporal_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }
        return (
            heatmaps.astype(np.float32, copy=False),
            details["temporal_weights"],
            logits.detach(),
            target_label,
            details,
        )
    finally:
        for capture in temporal_captures:
            capture.close()
        for module in spatial_modules + [pool_module]:
            module._chefer_attention = None


def generate_videoprism_heatmaps(model, pixel_values_videos, target_label=None):
    """Return raw Joint-ST heatmaps, frame relevance, logits, target, and details."""
    model.eval()
    if pixel_values_videos.dim() != 5 or pixel_values_videos.shape[0] != 1:
        raise ValueError("VideoPrism Chefer expects batch-size-one [B,T,C,H,W] input")
    video_model = model.video_model
    vision_model = video_model.vision_model
    spatial_captures = [
        AttentionCapture(layer.attention, f"VideoPrism spatial layer {layer_idx}")
        for (layer_idx, layer) in enumerate(vision_model.spatial_layers)
    ]
    temporal_captures = [
        AttentionCapture(layer.attention, f"VideoPrism temporal layer {layer_idx}")
        for (layer_idx, layer) in enumerate(vision_model.temporal_layers)
    ]
    auxiliary_captures = [
        AttentionCapture(layer.attention, f"VideoPrism auxiliary layer {layer_idx}")
        for (layer_idx, layer) in enumerate(video_model.auxiliary_layers)
    ]
    pool_capture = AttentionCapture(video_model.head, "VideoPrism attention pool")
    captures = (
        spatial_captures + temporal_captures + auxiliary_captures + [pool_capture]
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(pixel_values_videos)
        target_label = _resolve_target(logits, target_label)
        logits[0, target_label].backward()
        num_frames = pixel_values_videos.shape[1]
        tubelet = vision_model.config.tubelet_size
        grid_height = pixel_values_videos.shape[-2] // tubelet[1]
        grid_width = pixel_values_videos.shape[-1] // tubelet[2]
        if grid_height != grid_width:
            raise RuntimeError(
                f"VideoPrism patch grid must be square, got {grid_height}x{grid_width}"
            )
        num_patches = grid_height * grid_width
        pool_seed = pool_capture.cam()
        auxiliary_cams = [capture.cam() for capture in auxiliary_captures]
        output_token_relevance = (
            _propagate_rows(pool_seed, auxiliary_cams).squeeze(0).squeeze(0)
        )
        if output_token_relevance.numel() != num_frames * num_patches:
            raise RuntimeError(
                f"VideoPrism pool produced {output_token_relevance.numel()} token scores, expected {num_frames * num_patches}"
            )
        output_token_relevance = output_token_relevance.reshape(num_frames, num_patches)
        pre_spatial_relevance = output_token_relevance
        temporal_cams = [capture.cam() for capture in temporal_captures]
        for layer_idx, cam in enumerate(temporal_cams):
            if cam.shape[0] != num_patches:
                raise RuntimeError(
                    f"VideoPrism temporal layer {layer_idx} has {cam.shape[0]} patch batches, expected {num_patches}"
                )
        temporal_seed = output_token_relevance.transpose(0, 1).unsqueeze(1)
        pre_spatial_relevance = (
            _propagate_rows(temporal_seed, temporal_cams).squeeze(1).transpose(0, 1)
        )
        spatial_cams = [capture.cam() for capture in spatial_captures]
        for layer_idx, cam in enumerate(spatial_cams):
            if cam.shape[0] != num_frames:
                raise RuntimeError(
                    f"VideoPrism spatial layer {layer_idx} has {cam.shape[0]} frame batches, expected {num_frames}"
                )
        spatial_seed = pre_spatial_relevance.unsqueeze(1)
        patch_relevance = _propagate_rows(spatial_seed, spatial_cams).squeeze(1)
        spatial_raw = _reshape_patch_maps(patch_relevance, grid_height)
        spatial_raw_numpy = spatial_raw.detach().cpu().numpy().astype(np.float32)
        heatmaps = spatial_raw_numpy
        raw_temporal_weights = torch.ones(
            num_frames, dtype=torch.float32, device=pixel_values_videos.device
        )
        temporal_weights = raw_temporal_weights
        raw_temporal_weights = pre_spatial_relevance.sum(dim=1)
        temporal_weights = raw_temporal_weights
        heatmaps = spatial_raw_numpy
        details = {
            "spatial_raw": spatial_raw_numpy,
            "output_token_relevance": output_token_relevance.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "pre_spatial_token_relevance": pre_spatial_relevance.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "raw_temporal_weights": raw_temporal_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "temporal_weights": temporal_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }
        return (
            heatmaps.astype(np.float32, copy=False),
            details["temporal_weights"],
            logits.detach(),
            target_label,
            details,
        )
    finally:
        for capture in captures:
            capture.close()
