import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from joint_spatiotemporal_chefer import generate_joint_cls_relevance


class Captured:
    def __init__(self, cam):
        self.cam = cam[:, None]

    def get_attn(self):
        return self.cam

    def get_attn_gradients(self):
        return torch.ones_like(self.cam)


def test_joint_cls_matches_independent_dense_noncommuting_spatial_product():
    generator = torch.Generator().manual_seed(9)
    cams = [torch.rand(3, 5, 5, generator=generator) * 0.1 for _ in range(3)]
    spatial = torch.eye(5).expand(3, -1, -1).clone()
    for cam in cams:
        spatial = spatial + cam @ spatial
    joint = generate_joint_cls_relevance(
        list(enumerate(Captured(c) for c in cams)), 3, 2
    )
    np.testing.assert_allclose(
        3 * joint, spatial[:, 0, 1:].reshape(3, 2, 2).numpy(), rtol=1e-6
    )


def test_real_tiny_clip_spatial_and_joint_share_predictions_and_map_shape():
    from transformers import CLIPVisionConfig, CLIPVisionModel
    from chefer_off_the_shelf import generate_clip_heatmaps

    class Model(torch.nn.Module):
        temporal_aggregation = "mean"

        def __init__(self):
            super().__init__()
            config = CLIPVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
                attention_dropout=0,
                _attn_implementation="eager",
            )
            self.clip_model = torch.nn.Module()
            self.clip_model.vision_model = CLIPVisionModel(config).vision_model

        def forward(self, video):
            frames = video.permute(0, 2, 1, 3, 4).reshape(-1, 3, 4, 4)
            pooled = self.clip_model.vision_model(
                pixel_values=frames
            ).pooler_output.mean(0)
            score = (pooled * torch.arange(8)).sum().reshape(1)
            return torch.stack((score, -score), 1)

    torch.manual_seed(5)
    model = Model().eval()
    video = torch.randn(1, 3, 2, 4, 4)
    clean = model(video).detach()
    joint, logits, target = generate_clip_heatmaps(model, video, target_label=0)
    torch.testing.assert_close(clean, logits)
    assert target == 0
    assert np.max(joint) > 0
    dense = torch.eye(5).expand(2, -1, -1).clone()
    for layer in model.clip_model.vision_model.encoder.layers:
        captured = layer.self_attn
        cam = (
            (captured.get_attn().float() * captured.get_attn_gradients().float())
            .clamp(min=0)
            .mean(1)
        )
        dense = dense + cam @ dense
    np.testing.assert_allclose(
        2 * joint, dense[:, 0, 1:].reshape(2, 2, 2).numpy(), rtol=1e-5, atol=1e-8
    )


def test_real_tiny_siglip_spatial_and_joint_use_independent_readouts():
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from chefer_siglip import generate_per_frame_heatmaps_siglip

    class Model(torch.nn.Module):
        temporal_aggregation = "mean"

        def __init__(self):
            super().__init__()
            config = SiglipVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
                _attn_implementation="eager",
            )
            self.siglip_model = SiglipVisionModel(config)
            self.register_buffer("text_embeds", torch.randn(2, 8))

        def _aggregate_temporal_embeddings(self, frames):
            return frames.mean(1)

    torch.manual_seed(7)
    model = Model().eval()
    video = torch.randn(1, 3, 2, 4, 4)
    args = dict(target_label=0, device="cpu", patch_size=2)
    a = generate_per_frame_heatmaps_siglip(model, video, **args)
    b = generate_per_frame_heatmaps_siglip(
        model, video, propagation_mode="joint_spatiotemporal", **args
    )
    assert np.max(a) > 0
    normalized = (b - b.min(axis=(1, 2), keepdims=True)) / np.ptp(
        b, axis=(1, 2), keepdims=True
    )
    np.testing.assert_allclose(a, normalized, rtol=1e-5, atol=1e-7)
