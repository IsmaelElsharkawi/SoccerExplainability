"""Chefer / Chefer-T attribution for the off-the-shelf baselines of Table 1.

    --model clip        CLIP ViT-B/16 (image-language, standard Chefer per frame)
    --model peav        PE-AV base, video branch (Chefer-T)
    --model videoprism  VideoPrism LvT base (Chefer-T)

Needs a newer stack than the soccer models: transformers==5.22.2 and
timm==1.0.31 (the versions the hooks in attention_recorder.py mirror).
Checkpoints are pinned to the revisions the rebuttal runs used.

Every model sees the same 30 frames as the soccer models (the repo's
`read_frames_decord`, 'middle' sampling) and is fed the full frame squashed to
its square input, so its token grid covers the whole broadcast frame, as for
SigLIP, MatchVision and SoccerMaster. The explained target is the ground-truth
class logit (zero-shot: video embedding vs. class-name text embedding).

Output per clip, in the format `evaluate_grounding.py` reads:
    <clip>.npz   saliency_maps        (T, h, w)  Table 1 map (Chefer-T for video
                                                  models, Chefer for CLIP)
                 chefer_spatial_maps  (T, h, w)  per-frame min-max spatial map
                                                  (standard Chefer; Table 2)
                 temporal_weights     (T,)       w_t / max w (ones for CLIP)
                 frame_scores         (T,)       per-frame mean of saliency_maps
                 logits               (C,)
    <clip>.json  clip_id, prediction_text, ground_truth_text, provenance

Chefer-T for these architectures (paper Sec. 4.2, Alg. 1):
  * VideoPrism: spatial blocks per frame -> temporal blocks per patch -> two
    joint auxiliary blocks over all T*N tokens -> attention-pool query.
    The pool query's relevance is rolled back through the auxiliary blocks,
    split per frame and propagated through that frame's spatial rollout (Eq. 4).
    w_t averages the per-position temporal rollouts over positions and query
    frames (Eq. 5).
  * PE-AV: PE-Core ViT per frame with a latent attention-pool head -> temporal
    transformer with a CLS token whose output is the video embedding. The
    spatial map is the pool query's relevance through the frame's ViT rollout.
    The temporal transformer is read out through CLS rather than per position,
    so w_t is the CLS row of its rollout.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attention_recorder import (  # noqa: E402
    AttentionRecorder,
    hook_timm_attention,
    patch_hf_eager,
    patch_timm_attention_pool,
    rollout,
)

CHECKPOINTS = {
    "clip": ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "peav": ("facebook/pe-av-base", "1c4c329b78d80a18d860ebfb05daebff4bc44518"),
    "videoprism": ("google/videoprism-lvt-base-f16r288", "fb6de9f0eb7bc285be86bdca1cf7daa3e3ef51ff"),
}


def per_frame_minmax(maps):
    """Min-max normalize each frame's map to [0, 1] (zeros if constant)."""
    flat = maps.reshape(maps.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    span = hi - lo
    out = torch.where(span > 0, (flat - lo) / span.clamp(min=1e-12), torch.zeros_like(flat))
    return out.reshape(maps.shape)


def combine_two_stream(spatial, weights):
    """Paper Eq. 5: min-max spatial map scaled by w_t / max_t' w_t'."""
    spatial = per_frame_minmax(spatial)
    weights = weights / weights.max().clamp(min=1e-12)
    return spatial * weights[:, None, None], spatial, weights


def resize_frames(frames_uint8, size, mode):
    """(T, 3, H, W) uint8 -> (T, 3, size, size) float in [0, 1], full frame squashed."""
    frames = frames_uint8.float() / 255.0
    return F.interpolate(frames, size=(size, size), mode=mode, antialias=True, align_corners=False).clamp(0, 1)


class ClipAdapter:
    """CLIP ViT-B/16: mean of per-frame image embeddings vs. class-name text
    embeddings, like `SigLIP_Classifier`; standard Chefer on the CLS row."""

    method = "Chefer (per-frame spatial relevance, CLS row)"
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __init__(self, keywords, device, recorder, name, revision):
        from transformers import AutoTokenizer, CLIPModel
        from transformers.models.clip import modeling_clip

        self.device = device
        self.recorder = recorder
        self.model = CLIPModel.from_pretrained(name, revision=revision, attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        patch_hf_eager(modeling_clip, recorder)
        for i, layer in enumerate(self.model.vision_model.encoder.layers):
            recorder.track(layer.self_attn, f"spatial.{i}")

        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        with torch.no_grad():
            tokens = tokenizer(keywords, padding=True, return_tensors="pt").to(device)
            text = self.model.text_model(**tokens).pooler_output
            self.text_embeds = F.normalize(self.model.text_projection(text), dim=-1)
        self.size = self.model.config.vision_config.image_size
        self.grid = self.size // self.model.config.vision_config.patch_size
        # CLIP's learned softmax temperature (perturbation_faithfulness.py).
        self.logit_scale = self.model.logit_scale.exp().item()

    def preprocess(self, frames_uint8):
        """(T, 3, H, W) uint8 -> normalized (T, 3, size, size) on the device."""
        x = resize_frames(frames_uint8, self.size, "bicubic")
        return ((x - self.mean[:, None, None]) / self.std[:, None, None]).to(self.device)

    def forward_logits(self, x):
        """Class scores (C,) of a preprocessed clip (cosine similarities)."""
        pooled = self.model.vision_model(pixel_values=x).pooler_output
        image_embeds = F.normalize(self.model.visual_projection(pooled).mean(dim=0, keepdim=True), dim=-1)
        return (image_embeds @ self.text_embeds.t())[0]

    def explain(self, frames_uint8, target):
        x = self.preprocess(frames_uint8).requires_grad_(True)
        num_frames = x.shape[0]

        logits = self.forward_logits(x)
        logits[target].backward()

        cams = self.recorder.cams("spatial.")
        relevance = rollout(cams, cams[0].shape[-1], num_frames, self.device)
        spatial = relevance[:, 0, 1:].reshape(num_frames, self.grid, self.grid)
        spatial = per_frame_minmax(spatial)
        return {
            "saliency_maps": spatial,
            "chefer_spatial_maps": spatial,
            "temporal_weights": torch.ones(num_frames, device=self.device),
            "logits": logits.detach(),
        }


class VideoPrismAdapter:
    """VideoPrism LvT base: cosine similarity of the L2-normalized video and
    class-name text embeddings."""

    method = "Chefer-T (factorized spatial-temporal relevance)"

    def __init__(self, keywords, device, recorder, name, revision):
        from transformers import AutoTokenizer, VideoPrismClipModel
        from transformers.models.videoprism import modeling_videoprism

        self.device = device
        self.recorder = recorder
        self.model = VideoPrismClipModel.from_pretrained(name, revision=revision, attn_implementation="eager")
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        patch_hf_eager(modeling_videoprism, recorder)
        video = self.model.video_model
        for i, layer in enumerate(video.vision_model.spatial_layers):
            recorder.track(layer.attention, f"spatial.{i}")
        for i, layer in enumerate(video.vision_model.temporal_layers):
            recorder.track(layer.attention, f"temporal.{i}")
        for i, layer in enumerate(video.auxiliary_layers):
            recorder.track(layer.attention, f"joint.{i}")
        recorder.track(video.head, "pool")

        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        with torch.no_grad():
            tokens = tokenizer(keywords, padding="max_length", max_length=64, truncation=True,
                               return_tensors="pt").to(device)
            text = self.model.text_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
            self.text_embeds = text.pooler_output.reshape(len(keywords), -1)
        config = self.model.config.vision_config
        self.size = config.image_size if isinstance(config.image_size, int) else config.image_size[0]
        self.grid = self.size // config.tubelet_size[1]
        # VideoPrismClipModel.forward exponentiates the raw similarities
        # (temperature 1), so that is the temperature of its own probabilities.
        self.logit_scale = 1.0

    def preprocess(self, frames_uint8):
        """(T, 3, H, W) uint8 -> (T, 3, size, size) in [0, 1] on the device."""
        return resize_frames(frames_uint8, self.size, "bilinear").to(self.device)

    def forward_logits(self, x):
        """Class scores (C,) of a preprocessed clip."""
        video = self.model.video_model(pixel_values_videos=x[None], interpolate_pos_encoding=True)
        video_embeds = video.pooler_output.reshape(1, -1)
        return (video_embeds @ self.text_embeds.t())[0]

    def explain(self, frames_uint8, target):
        x = self.preprocess(frames_uint8).requires_grad_(True)
        num_frames, num_patches = x.shape[0], self.grid * self.grid

        logits = self.forward_logits(x)
        logits[target].backward()

        # Per-frame spatial rollout: cams (T, N, N).
        spatial_cams = self.recorder.cams("spatial.")
        r_spat = rollout(spatial_cams, num_patches, num_frames, self.device)
        # Pool query -> joint auxiliary blocks over T*N tokens (frame-major).
        joint_cams = self.recorder.cams("joint.")
        r_joint = rollout(joint_cams, num_frames * num_patches, 1, self.device)
        pool = self.recorder.cams("pool")[0]  # (1, 1, T*N)
        token_relevance = torch.bmm(pool, r_joint)[0, 0].reshape(num_frames, 1, num_patches)
        spatial = torch.bmm(token_relevance, r_spat)[:, 0].reshape(num_frames, self.grid, self.grid)
        # Per-position temporal rollout: cams (N, T, T); Eq. 5 averages over p and t'.
        temporal_cams = self.recorder.cams("temporal.")
        r_temp = rollout(temporal_cams, num_frames, num_patches, self.device)
        weights = r_temp.mean(dim=(0, 1))

        final, spatial, weights = combine_two_stream(spatial, weights)
        return {
            "saliency_maps": final,
            "chefer_spatial_maps": spatial,
            "temporal_weights": weights,
            "logits": logits.detach(),
        }


def load_pe_video_branch(name, revision):
    """The video-text branch of a PE-AV checkpoint as a `PeVideoModel`.

    `PeAudioVideoModel.from_pretrained` cannot load facebook/pe-av-base under
    transformers 5.22.2: the nested video config is rebuilt without its 1024
    timm label names (so the PE-Core head comes out 2-wide), and an audio
    weight-tying step crashes. The audio side is not needed here, so build
    `PeVideoModel` from the checkpoint's own video config and load its
    weights by hand: `video_model.*`, plus the shared `text_model.*` that
    `PeAudioVideoModel` ties into `video_model.text_model`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, PeVideoModel

    weights_path = hf_hub_download(name, "model.safetensors", revision=revision)
    state = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("video_model."):
                state[key[len("video_model."):]] = f.get_tensor(key)
            elif key.startswith("text_model."):
                state[key] = f.get_tensor(key)

    config = AutoConfig.from_pretrained(name, revision=revision).video_config
    head = state["video_encoder.embedder.vision_model.timm_model.head.weight"]
    config.video_config.vision_config.num_labels = head.shape[0]
    model = PeVideoModel._from_config(config, attn_implementation="eager")

    result = model.load_state_dict(state, strict=False)
    # Rotary inverse frequencies are recomputed buffers, never stored.
    missing = [k for k in result.missing_keys if not k.endswith("inv_freq")]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"PE-AV video branch did not load cleanly: missing={missing[:10]} "
                           f"({len(missing)} total), unexpected={result.unexpected_keys[:10]}")
    print(f"Loaded PE-AV video branch: {len(state)} tensors, head {tuple(head.shape)}")
    return model


class PeAvAdapter:
    """PE-AV base, video-text branch of `PeVideoModel.forward`: the temporal
    CLS embedding through `video_head` against `text_video_head` text
    embeddings. The explained score is the raw dot product; the learned
    logit scale/bias are left out because HF applies the scale as a plain
    multiplier, and a negative value would flip the sign of every gradient
    Chefer keeps."""

    method = "Chefer-T (factorized spatial-temporal relevance)"
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    def __init__(self, keywords, device, recorder, name, revision):
        from transformers import AutoTokenizer
        from transformers.models.pe_video import modeling_pe_video

        self.device = device
        self.recorder = recorder
        self.model = load_pe_video_branch(name, revision).to(device).eval()
        self.model.requires_grad_(False)

        patch_hf_eager(modeling_pe_video, recorder)
        for i, layer in enumerate(self.model.video_encoder.layers):
            recorder.track(layer.self_attn, f"temporal.{i}")
        vit = self.model.video_encoder.embedder.vision_model.timm_model
        self.hooks = [hook_timm_attention(block.attn, f"spatial.{i}", recorder) for i, block in enumerate(vit.blocks)]
        patch_timm_attention_pool(vit.attn_pool, "pool", recorder)
        self.num_prefix = vit.num_prefix_tokens
        self.grid = vit.patch_embed.grid_size[0]
        self.size = vit.patch_embed.img_size[0]

        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        with torch.no_grad():
            tokens = tokenizer(keywords, padding=True, return_tensors="pt").to(device)
            text = self.model.text_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                                         output_hidden_states=True)
            self.text_embeds = self.model.text_video_head(text.hidden_states[-1][:, 0])
        # PeVideoModel.forward applies this as a plain multiplier (no exp);
        # the bias cancels in a softmax over classes.
        self.logit_scale = float(self.model.text_video_logit_scale.reshape(-1)[0].item())

    def preprocess(self, frames_uint8):
        """(T, 3, H, W) uint8 -> normalized (T, 3, size, size) on the device."""
        x = resize_frames(frames_uint8, self.size, "bilinear")
        return ((x - self.mean[:, None, None]) / self.std[:, None, None]).to(self.device)

    def forward_logits(self, x):
        """Class scores (C,) of a preprocessed clip (raw dot products, see above)."""
        video = self.model.video_encoder(pixel_values_videos=x[None])
        video_embeds = self.model.video_head(video.pooler_output)
        return (video_embeds @ self.text_embeds.t())[0]

    def explain(self, frames_uint8, target):
        x = self.preprocess(frames_uint8).requires_grad_(True)
        num_frames = x.shape[0]

        logits = self.forward_logits(x)
        logits[target].backward()

        spatial_cams = self.recorder.cams("spatial.")  # (T, 1+N, 1+N)
        r_spat = rollout(spatial_cams, spatial_cams[0].shape[-1], num_frames, self.device)
        pool = self.recorder.cams("pool")[0]  # (T, 1, 1+N)
        spatial = torch.bmm(pool, r_spat)[:, 0, self.num_prefix:].reshape(num_frames, self.grid, self.grid)
        temporal_cams = self.recorder.cams("temporal.")  # (1, 1+T, 1+T), CLS first
        r_temp = rollout(temporal_cams, num_frames + 1, 1, self.device)
        weights = r_temp[0, 0, 1:]

        final, spatial, weights = combine_two_stream(spatial, weights)
        return {
            "saliency_maps": final,
            "chefer_spatial_maps": spatial,
            "temporal_weights": weights,
            "logits": logits.detach(),
        }


ADAPTERS = {"clip": ClipAdapter, "peav": PeAvAdapter, "videoprism": VideoPrismAdapter}


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def sanitize(clip_id):
    """Same stem as `inference_utils._sanitize_clip_id`."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(clip_id))


def iter_clips(test_config):
    for json_path, video_base in zip(test_config["json"], test_config["video_base"]):
        with open(json_path) as f:
            items = json.load(f)
        for item in items:
            variant = item.get("variant")
            clip_id = item["video"] + (f"#{variant}" if variant else "")
            yield clip_id, os.path.join(video_base, item["video"]), item["caption"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--config_path", required=True, help="Same config as the soccer-model scripts.")
    parser.add_argument("--saliency_save_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--checkpoint", default=None, help="Override the pinned Hub checkpoint (debugging).")
    parser.add_argument("--revision", default=None, help="Override the pinned checkpoint revision.")
    parser.add_argument("--max_clips", type=int, default=None, help="Debug: stop after this many clips.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute clips that already have artifacts.")
    args = parser.parse_args()

    from dataset.video_utils_siglip import read_frames_decord

    config = load_config(args.config_path)["dataset"]["test"]
    keywords = config["keywords"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.saliency_save_dir, exist_ok=True)

    name, revision = CHECKPOINTS[args.model]
    if args.checkpoint:
        name, revision = args.checkpoint, args.revision
    recorder = AttentionRecorder()
    adapter = ADAPTERS[args.model](keywords, device, recorder, name, revision)
    print(f"[{args.model}] {name}@{(revision or 'main')[:8]} on {device}; {adapter.method}")

    n_done = n_correct = 0
    started = time.time()
    for index, (clip_id, video_path, caption) in enumerate(iter_clips(config)):
        if args.max_clips is not None and index >= args.max_clips:
            break
        stem = sanitize(clip_id)
        npz_path = os.path.join(args.saliency_save_dir, stem + ".npz")
        json_path = os.path.join(args.saliency_save_dir, stem + ".json")
        if not args.overwrite and os.path.isfile(npz_path) and os.path.isfile(json_path):
            continue
        if caption not in keywords:
            raise ValueError(f"{clip_id}: caption {caption!r} is not one of the configured keywords")

        # No silent substitution: a clip that fails to decode stops the run
        # (VideoCaptionDataset would swap in a random other clip instead).
        frames, frame_indices, _ = read_frames_decord(video_path, args.num_frames, "middle")
        target = keywords.index(caption)

        recorder.reset()
        recorder.enabled = True
        out = adapter.explain(frames, target)
        recorder.enabled = False
        recorder.reset()

        maps = out["saliency_maps"].detach().float().cpu().numpy()
        logits = out["logits"].float().cpu().numpy()
        prediction = keywords[int(np.argmax(logits))]
        np.savez_compressed(
            npz_path,
            saliency_maps=maps,
            chefer_spatial_maps=out["chefer_spatial_maps"].detach().float().cpu().numpy(),
            temporal_weights=out["temporal_weights"].detach().float().cpu().numpy(),
            frame_scores=maps.reshape(maps.shape[0], -1).mean(axis=1),
            logits=logits,
            frame_indices=np.asarray(frame_indices),
        )
        with open(json_path, "w") as f:
            json.dump({
                "clip_id": clip_id,
                "video_name": os.path.basename(video_path),
                "num_frames": int(maps.shape[0]),
                "prediction_text": prediction,
                "ground_truth_text": caption,
                "target_index": target,
                "model": args.model,
                "checkpoint": name,
                "checkpoint_revision": revision,
                "method": adapter.method,
                "map_grid": list(maps.shape[1:]),
            }, f, indent=2)

        n_done += 1
        n_correct += prediction == caption
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{index + 1}] {clip_id}: gt={caption!r} pred={prediction!r} "
              f"acc so far {100 * n_correct / n_done:.1f}% ({time.time() - started:.0f}s)")

    print(f"[{args.model}] wrote {n_done} clips to {args.saliency_save_dir}")


if __name__ == "__main__":
    main()
