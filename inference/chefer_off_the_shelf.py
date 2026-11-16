"""Joint-ST attribution for independent CLIP frame encoders."""

from inference_utils import wrap_clip_attention_module


def _resolve_target(logits, target_label):
    if logits.shape[0] != 1:
        raise ValueError(
            f"Chefer inference currently requires batch size 1, got {logits.shape[0]}"
        )
    if target_label is None:
        return int(logits[0].argmax().item())
    target_label = int(target_label)
    if not 0 <= target_label < logits.shape[1]:
        raise ValueError(
            f"Target label {target_label} is outside [0, {logits.shape[1]})"
        )
    return target_label


def generate_clip_heatmaps(model, video_frames, target_label=None):
    """Return raw Joint-ST maps for a mean-pooled CLIP video classifier."""
    model.eval()
    clip_vision = model.clip_model.vision_model
    spatial_modules = []
    for layer_idx, layer in enumerate(clip_vision.encoder.layers):
        spatial_modules.append((layer_idx, wrap_clip_attention_module(layer.self_attn)))
    if video_frames.dim() != 5:
        raise ValueError(
            f"CLIP video input must be [B,C,T,H,W], got {video_frames.shape}"
        )
    if video_frames.shape[0] != 1:
        raise ValueError("CLIP Chefer supports batch-size-one inference")
    num_frames = video_frames.shape[2]
    grid_size = clip_vision.config.image_size // clip_vision.config.patch_size
    model.zero_grad(set_to_none=True)
    logits = model(video_frames)
    target_label = _resolve_target(logits, target_label)
    logits[0, target_label].backward()
    from joint_spatiotemporal_chefer import generate_joint_cls_relevance

    if getattr(model, "temporal_aggregation", "mean") != "mean":
        raise ValueError("Joint CLIP readout requires mean frame aggregation")
    heatmaps = generate_joint_cls_relevance(spatial_modules, num_frames, grid_size)
    return (heatmaps, logits.detach(), target_label)
