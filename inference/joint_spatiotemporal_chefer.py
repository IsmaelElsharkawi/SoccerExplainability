"""Factorized joint spatiotemporal Chefer propagation.

This module propagates a single relevance field over the video's T x N patch
tokens without materializing a (T*N) x (T*N) matrix.  Each attention residual
uses the Chefer transition ``I + mean_heads(clamp(grad * attention, min=0))``.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
import torch


WrappedAttention = Iterable[Tuple[int, torch.nn.Module]]


def _require_attention_pair(module, description):
    attention = module.get_attn()
    gradient = module.get_attn_gradients()
    if attention is None or gradient is None:
        raise RuntimeError(
            f"Missing captured attention or gradient for {description}. "
            "The joint propagation requires every wrapped attention transition."
        )
    if attention.shape != gradient.shape:
        raise ValueError(
            f"Attention/gradient shape mismatch for {description}: "
            f"{tuple(attention.shape)} vs {tuple(gradient.shape)}"
        )
    return attention, gradient


def _mean_positive_gradient_attention(module, description):
    """Compute mean-head positive grad-attention with bounded GPU memory."""
    attention, gradient = _require_attention_pair(module, description)
    if attention.ndim != 4:
        raise ValueError(
            f"{description} has rank {attention.ndim}; expected [B,H,Q,K]"
        )

    # SoccerMaster's spatial tensor is approximately 1 GiB per layer in
    # float16.  Converting all heads and the product to float32 simultaneously
    # can exceed even an A100.  Accumulating one head at a time is algebraically
    # identical to mean(dim=1) and caps temporary memory at one [B,Q,K] slice.
    cam = torch.zeros(
        attention.shape[0],
        attention.shape[2],
        attention.shape[3],
        device=attention.device,
        dtype=torch.float32,
    )
    for head_idx in range(attention.shape[1]):
        head_cam = (
            gradient[:, head_idx].float()
            * attention[:, head_idx].float()
        ).clamp(min=0)
        cam.add_(head_cam)
        del head_cam
    return cam.div_(attention.shape[1])


def _head_frame_seed(head_layers: WrappedAttention, num_frames: int, device):
    """Propagate the classifier's uniform temporal mean-pool readout."""
    relevance = torch.full(
        (num_frames,),
        1.0 / num_frames,
        device=device,
        dtype=torch.float32,
    )
    for layer_idx, module in reversed(list(head_layers)):
        cam = _mean_positive_gradient_attention(
            module,
            f"classifier temporal layer {layer_idx}",
        )
        if cam.shape[-2:] != (num_frames, num_frames):
            raise ValueError(
                f"Classifier temporal layer {layer_idx} has shape "
                f"{tuple(cam.shape)}, expected [...,{num_frames},{num_frames}]"
            )
        cam = cam.reshape(-1, num_frames, num_frames).mean(dim=0)
        relevance = relevance + torch.matmul(relevance, cam)
    return relevance


def _pooling_seed(
    pooling_module,
    frame_relevance,
    num_frames: int,
    num_patches: int,
):
    """Map frame-level probe relevance to the final backbone patch tokens."""
    cam = _mean_positive_gradient_attention(
        pooling_module,
        "pooling cross-attention",
    )
    expected = (num_frames, num_patches)
    if cam.shape[0] != num_frames or cam.shape[-2:] != (1, num_patches):
        raise ValueError(
            f"Pooling cross-attention has shape {tuple(cam.shape)}, expected "
            f"[{num_frames},1,{num_patches}] after head averaging"
        )
    cam = cam.squeeze(-2)
    if tuple(cam.shape) != expected:
        raise ValueError(
            f"Reduced pooling cross-attention has shape {tuple(cam.shape)}, "
            f"expected {expected}"
        )
    return frame_relevance.unsqueeze(1) * cam


def _apply_spatial_transition(relevance, module, layer_idx, num_frames, num_patches):
    """Apply a block-diagonal spatial ``I + A`` transition to [T,N]."""
    cam = _mean_positive_gradient_attention(
        module,
        f"spatial layer {layer_idx}",
    )
    if cam.shape[0] != num_frames or cam.shape[-2:] != (num_patches, num_patches):
        raise ValueError(
            f"Spatial layer {layer_idx} has shape {tuple(cam.shape)}, expected "
            f"[{num_frames},{num_patches},{num_patches}] after head averaging"
        )
    return relevance + torch.bmm(relevance.unsqueeze(1), cam).squeeze(1)


def _apply_temporal_transition(relevance, module, layer_idx, num_frames, num_patches):
    """Apply a patch-block-diagonal temporal ``I + A`` transition to [T,N]."""
    cam = _mean_positive_gradient_attention(
        module,
        f"temporal layer {layer_idx}",
    )
    if cam.shape[0] != num_patches or cam.shape[-2:] != (num_frames, num_frames):
        raise ValueError(
            f"Temporal layer {layer_idx} has shape {tuple(cam.shape)}, expected "
            f"[{num_patches},{num_frames},{num_frames}] after head averaging"
        )
    relevance_by_patch = relevance.transpose(0, 1)
    relevance_by_patch = relevance_by_patch + torch.bmm(
        relevance_by_patch.unsqueeze(1),
        cam,
    ).squeeze(1)
    return relevance_by_patch.transpose(0, 1)


def generate_joint_spatiotemporal_relevance(
    wrapped: Dict[str, object],
    num_frames: int,
    patch_size: int,
) -> np.ndarray:
    """Return one raw, nonnegative Chefer-ST volume with shape [T,H,W].

    The forward block order is temporal attention followed by spatial
    attention.  Relevance therefore traverses each block in reverse:
    spatial, then temporal.  Spatial-only blocks simply omit the second step.
    No min-max normalization, clamping, or post-hoc temporal weighting is
    applied to the returned volume.
    """
    if num_frames <= 0 or patch_size <= 0:
        raise ValueError("num_frames and patch_size must be positive")

    num_patches = patch_size * patch_size
    spatial_layers = list(wrapped["spatial_attn"])
    temporal_by_layer = dict(wrapped["temporal_attn"])
    if not spatial_layers:
        raise ValueError("At least one spatial attention layer is required")

    first_attention, _ = _require_attention_pair(
        spatial_layers[0][1],
        f"spatial layer {spatial_layers[0][0]}",
    )
    device = first_attention.device

    frame_relevance = _head_frame_seed(
        wrapped["head_te_attn"],
        num_frames=num_frames,
        device=device,
    )
    relevance = _pooling_seed(
        wrapped["pooling_head_attn"],
        frame_relevance=frame_relevance,
        num_frames=num_frames,
        num_patches=num_patches,
    )

    for layer_idx, spatial_module in reversed(spatial_layers):
        relevance = _apply_spatial_transition(
            relevance,
            spatial_module,
            layer_idx,
            num_frames,
            num_patches,
        )
        temporal_module = temporal_by_layer.get(layer_idx)
        if temporal_module is not None:
            relevance = _apply_temporal_transition(
                relevance,
                temporal_module,
                layer_idx,
                num_frames,
                num_patches,
            )

    if not torch.isfinite(relevance).all():
        raise FloatingPointError("Joint spatiotemporal relevance contains non-finite values")
    if (relevance < 0).any():
        raise FloatingPointError("Joint spatiotemporal relevance unexpectedly became negative")

    heatmaps = relevance.detach().cpu().numpy().astype(np.float32, copy=False)
    return heatmaps.reshape(num_frames, patch_size, patch_size)


def generate_joint_cls_relevance(spatial_layers, num_frames, patch_size):
    """Joint-token rollout for independent CLS-based frame encoders.

    The frame-mean readout seeds each final CLS with 1/T. Each spatial
    transition is a block of the joint frame-token graph. There are no
    cross-frame attention edges to add for CLIP. This reverse row rollout is
    computed independently of the forward dense spatial accumulator.
    """
    layers = list(spatial_layers)
    if not layers or num_frames <= 0 or patch_size <= 0:
        raise ValueError("Require spatial attention and positive frame/grid sizes")
    attention, _ = _require_attention_pair(layers[0][1], "first spatial layer")
    tokens = patch_size * patch_size + 1
    relevance = torch.zeros(num_frames, tokens, device=attention.device, dtype=torch.float32)
    relevance[:, 0] = 1.0 / num_frames
    for index, module in reversed(layers):
        relevance = _apply_spatial_transition(relevance, module, index, num_frames, tokens)
    if not torch.isfinite(relevance).all() or (relevance < 0).any():
        raise FloatingPointError("Invalid joint CLS relevance")
    return relevance[:, 1:].detach().cpu().numpy().reshape(num_frames, patch_size, patch_size)
