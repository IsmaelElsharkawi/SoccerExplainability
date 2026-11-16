"""Small executable tests for foundation-video Chefer propagation."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "inference")),
)


from chefer_video_foundation import (
    _propagate_rows,
    generate_peav_heatmaps,
    generate_videoprism_heatmaps,
)


class _SelfAttention(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, hidden_states):
        batch, tokens, dim = hidden_states.shape

        def split(projection):
            return projection.reshape(
                batch,
                tokens,
                self.heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split(self.q(hidden_states)) * self.head_dim**-0.5
        key = split(self.k(hidden_states))
        value = split(self.v(hidden_states))
        attention = (query @ key.transpose(-2, -1)).softmax(dim=-1)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.out(output), attention


class _Layer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attention = _SelfAttention(dim=dim)

    def forward(self, hidden_states):
        output, _ = self.attention(hidden_states)
        return hidden_states + output


class _Pool(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, hidden_states):
        batch, tokens, dim = hidden_states.shape
        query = (
            self.q(self.query.expand(batch, -1, -1))
            .reshape(batch, 1, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k(hidden_states)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v(hidden_states)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        attention = ((query * self.head_dim**-0.5) @ key.transpose(-2, -1)).softmax(
            dim=-1
        )
        output = attention @ value
        return output.transpose(1, 2).reshape(batch, 1, dim), attention


class _VisionModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.config = SimpleNamespace(tubelet_size=(1, 1, 1))
        self.spatial_layers = nn.ModuleList([_Layer(dim), _Layer(dim)])
        self.temporal_layers = nn.ModuleList([_Layer(dim), _Layer(dim)])


class _VideoModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.vision_model = _VisionModel(dim)
        self.auxiliary_layers = nn.ModuleList([_Layer(dim), _Layer(dim)])
        self.head = _Pool(dim)


class _Classifier(nn.Module):
    def __init__(self, dim=8, classes=3):
        super().__init__()
        self.video_model = _VideoModel(dim)
        self.input_projection = nn.Linear(3, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, pixel_values):
        batch, frames, channels, height, width = pixel_values.shape
        hidden = pixel_values.permute(0, 1, 3, 4, 2)
        hidden = self.input_projection(hidden)
        patches = height * width
        hidden = hidden.reshape(batch * frames, patches, -1)
        for layer in self.video_model.vision_model.spatial_layers:
            hidden = layer(hidden)

        hidden = hidden.reshape(batch, frames, patches, -1)
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch * patches,
            frames,
            -1,
        )
        for layer in self.video_model.vision_model.temporal_layers:
            hidden = layer(hidden)

        hidden = hidden.reshape(batch, patches, frames, -1)
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch,
            frames * patches,
            -1,
        )
        for layer in self.video_model.auxiliary_layers:
            hidden = layer(hidden)
        pooled, _ = self.video_model.head(hidden)
        return self.classifier(pooled[:, 0])

    @staticmethod
    def get_types(logits):
        return logits.argmax(dim=1, keepdim=True)


def test_row_propagation_matches_full_rollout():
    torch.manual_seed(7)
    seed = torch.rand(3, 2, 5)
    cams = [torch.rand(3, 5, 5) for _ in range(4)]
    rollout = torch.eye(5).expand(3, -1, -1).clone()
    for cam in cams:
        rollout = rollout + torch.bmm(cam, rollout)
    torch.testing.assert_close(
        _propagate_rows(seed, cams),
        torch.bmm(seed, rollout),
    )


def test_videoprism_joint_path_returns_raw_propagated_volume():
    torch.manual_seed(13)
    model = _Classifier()
    inputs = torch.rand(1, 3, 3, 2, 2)

    heatmaps, weights, logits, target, details = generate_videoprism_heatmaps(
        model,
        inputs,
        target_label=1,
    )

    assert heatmaps.shape == (3, 2, 2)
    assert weights.shape == (3,)
    assert logits.shape == (1, 3)
    assert target == 1
    assert np.isfinite(heatmaps).all()
    assert (heatmaps >= 0).all()
    np.testing.assert_allclose(heatmaps, details["spatial_raw"])


def test_peav_joint_path_preserves_classifier_and_raw_frame_relevance():
    import timm.layers as layers

    if not hasattr(layers, "resolve_self_attn_mask"):
        pytest.skip("PE-AV requires the timm version documented in docs/joint_st.md")
    from timm.models.eva import EvaAttention

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 8)
            self.spatial_encoder = nn.Module()
            self.spatial_encoder.blocks = nn.ModuleList([nn.Module(), nn.Module()])
            for block in self.spatial_encoder.blocks:
                block.attn = EvaAttention(8, num_heads=2, norm_layer=nn.LayerNorm)
            self.spatial_encoder.attn_pool = layers.AttentionPoolLatent(8, num_heads=2)
            self.spatial_encoder.num_prefix_tokens = 0
            self.spatial_encoder.patch_embed = SimpleNamespace(grid_size=(2, 2))
            self.temporal_encoder = nn.Module()
            self.temporal_encoder.layers = nn.ModuleList([nn.Module()])
            self.temporal_encoder.layers[0].self_attn = _SelfAttention()
            self.cls = nn.Parameter(torch.randn(1, 1, 8))
            self.classifier = nn.Linear(8, 3)

        def forward(self, frames):
            hidden = self.projection(frames[0].permute(0, 2, 3, 1).reshape(3, 4, 3))
            for block in self.spatial_encoder.blocks:
                hidden = hidden + block.attn(hidden)
            pooled = self.spatial_encoder.attn_pool(hidden).unsqueeze(0)
            tokens = torch.cat([self.cls, pooled], dim=1)
            for layer in self.temporal_encoder.layers:
                tokens = tokens + layer.self_attn(tokens)[0]
            return self.classifier(tokens[:, 0])

    torch.manual_seed(19)
    model = Model().eval()
    frames = torch.randn(1, 3, 3, 2, 2)
    clean = model(frames).detach()
    maps, weights, logits, target, details = generate_peav_heatmaps(
        model, frames, target_label=1
    )
    torch.testing.assert_close(logits, clean, rtol=1e-5, atol=1e-6)
    assert target == 1
    assert maps.shape == (3, 2, 2)
    assert np.isfinite(maps).all() and (maps >= 0).all()
    np.testing.assert_allclose(maps, details["spatial_raw"] * weights[:, None, None])
    assert not model.temporal_encoder.layers[0].self_attn._forward_hooks


def test_joint_st_runner_explains_manifest_class_and_saves_variant(
    tmp_path, monkeypatch
):
    import json
    import run_joint_st as runner

    model = _Classifier().eval()
    model.preprocess = lambda raw: raw.float() / 255
    monkeypatch.setattr(runner, "load_model", lambda *args: model)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path: {
            "dataset": {
                "test": {"keywords": ["goal", "corner", "foul"], "sample": "middle"}
            }
        },
    )
    raw = torch.randint(0, 255, (3, 3, 2, 2), dtype=torch.uint8)
    monkeypatch.setattr(
        runner, "read_frames_decord", lambda *args: (raw, [0, 4, 8], 1.0)
    )
    manifest = tmp_path / "clips.json"
    manifest.write_text(
        json.dumps([{"video": "match/1.mp4", "caption": "corner", "variant": "corner"}])
    )
    output = tmp_path / "maps"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_joint_st.py",
            "--model",
            "videoprism",
            "--config_path",
            "unused.py",
            "--test_json",
            str(manifest),
            "--video_base",
            str(tmp_path),
            "--output_dir",
            str(output),
            "--device",
            "cpu",
            "--num_frames",
            "3",
        ],
    )
    runner.main()
    metadata = json.loads((output / "match_1_mp4_corner.json").read_text())
    assert metadata["clip_id"] == "match/1.mp4#corner"
    assert metadata["target_index"] == 1
    with np.load(output / "match_1_mp4_corner.npz") as saved:
        assert saved["saliency_maps"].shape == (3, 2, 2)
        np.testing.assert_array_equal(saved["frame_indices"], [0, 4, 8])
