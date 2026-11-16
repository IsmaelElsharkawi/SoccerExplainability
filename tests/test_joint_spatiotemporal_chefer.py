import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "inference"))

from joint_spatiotemporal_chefer import generate_joint_spatiotemporal_relevance


class CapturedAttention:
    def __init__(self, cam):
        self._attention = cam.clone()
        self._gradient = torch.ones_like(cam)

    def get_attn(self):
        return self._attention

    def get_attn_gradients(self):
        return self._gradient


def _spatial_dense(cam, num_frames, num_patches):
    dense = torch.zeros(num_frames * num_patches, num_frames * num_patches)
    for frame_idx in range(num_frames):
        start = frame_idx * num_patches
        dense[start:start + num_patches, start:start + num_patches] = (
            torch.eye(num_patches) + cam[frame_idx]
        )
    return dense


def _temporal_dense(cam, num_frames, num_patches):
    dense = torch.eye(num_frames * num_patches)
    for patch_idx in range(num_patches):
        for output_frame in range(num_frames):
            output_idx = output_frame * num_patches + patch_idx
            for input_frame in range(num_frames):
                input_idx = input_frame * num_patches + patch_idx
                dense[output_idx, input_idx] += cam[
                    patch_idx,
                    output_frame,
                    input_frame,
                ]
    return dense


def test_factorized_propagation_matches_dense_joint_graph():
    generator = torch.Generator().manual_seed(7)
    num_frames = 3
    patch_size = 2
    num_patches = patch_size**2

    spatial = {
        layer_idx: 0.05 * torch.rand(
            num_frames,
            num_patches,
            num_patches,
            generator=generator,
        )
        for layer_idx in range(2)
    }
    temporal = {
        layer_idx: 0.05 * torch.rand(
            num_patches,
            num_frames,
            num_frames,
            generator=generator,
        )
        for layer_idx in range(2)
    }
    head = 0.05 * torch.rand(
        num_frames,
        num_frames,
        generator=generator,
    )
    pooling = 0.05 * torch.rand(
        num_frames,
        num_patches,
        generator=generator,
    )

    wrapped = {
        "spatial_attn": [
            (idx, CapturedAttention(cam.unsqueeze(1)))
            for idx, cam in spatial.items()
        ],
        "temporal_attn": [
            (idx, CapturedAttention(cam.unsqueeze(1)))
            for idx, cam in temporal.items()
        ],
        "pooling_head_attn": CapturedAttention(
            pooling[:, None, None, :]
        ),
        "head_te_attn": [
            (0, CapturedAttention(head[None, None, :, :]))
        ],
    }

    actual = generate_joint_spatiotemporal_relevance(
        wrapped,
        num_frames=num_frames,
        patch_size=patch_size,
    ).reshape(-1)

    frame_seed = torch.full((num_frames,), 1.0 / num_frames)
    frame_seed = frame_seed @ (torch.eye(num_frames) + head)
    expected = (frame_seed[:, None] * pooling).reshape(-1)
    for layer_idx in reversed(range(2)):
        expected = expected @ _spatial_dense(
            spatial[layer_idx],
            num_frames,
            num_patches,
        )
        expected = expected @ _temporal_dense(
            temporal[layer_idx],
            num_frames,
            num_patches,
        )

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-7)


def test_zero_self_attention_retains_identity_and_raw_pooling_scale():
    num_frames = 2
    patch_size = 2
    num_patches = patch_size**2
    pooling = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    zero_spatial = torch.zeros(num_frames, 1, num_patches, num_patches)
    zero_temporal = torch.zeros(num_patches, 1, num_frames, num_frames)

    wrapped = {
        "spatial_attn": [(0, CapturedAttention(zero_spatial))],
        "temporal_attn": [(0, CapturedAttention(zero_temporal))],
        "pooling_head_attn": CapturedAttention(pooling[:, None, None, :]),
        "head_te_attn": [],
    }
    actual = generate_joint_spatiotemporal_relevance(
        wrapped,
        num_frames=num_frames,
        patch_size=patch_size,
    )

    expected = (pooling / num_frames).reshape(num_frames, patch_size, patch_size)
    torch.testing.assert_close(torch.from_numpy(actual), expected)


def test_headwise_accumulation_equals_float32_mean():
    from joint_spatiotemporal_chefer import _mean_positive_gradient_attention

    attention = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ],
        dtype=torch.float16,
    )
    gradient = torch.tensor(
        [
            [
                [[1.0, -1.0], [0.5, 2.0]],
                [[-1.0, 0.25], [2.0, 0.5]],
            ]
        ],
        dtype=torch.float16,
    )
    captured = CapturedAttention(attention)
    captured._gradient = gradient

    actual = _mean_positive_gradient_attention(captured, "test")
    expected = (attention.float() * gradient.float()).clamp(min=0).mean(dim=1)
    torch.testing.assert_close(actual, expected)
