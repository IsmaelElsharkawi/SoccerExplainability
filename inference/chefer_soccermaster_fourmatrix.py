"""
Chefer explainability inference for SoccerMaster classification

SoccerMaster uses SigLIP2-large-patch16-512 as its vision backbone, with temporal+spatial
attention starting at layer 16 (of 27 total). The temporal attention is excluded from the
relevance propagation. The CaptionClassificationHead performs event classification into the
same 23 SoccerNet-v2 event classes as MatchVision.

Architecture:
    VisionBackbone (SigLIP2-large) -> CaptionClassificationHead
    - 512x512 input -> 32x32 = 1024 patches, 1024-dim hidden
    - Layers 0-15: spatial attention only
    - Layers 16-26: temporal + spatial attention
    - Pooling head: learnable probe cross-attention (same as SigLIP)
    - Classification: LN -> TransformerEncoder(2 layers) -> avg_pool -> LN -> Linear

================================================================================
CHEFER METHOD
================================================================================

Based on:
    "Generic Attention-model Explainability for Interpreting Bi-Modal and
     Encoder-Decoder Transformers" (Chefer et al., ICCV 2021)

1. Forward pass through VisionBackbone + CaptionClassificationHead
   - Phase 1 (layers 0-15): spatial attention only
   - Inject temporal positional embedding
   - Phase 2 (layers 16-26): temporal + spatial attention per block
   - Pooling head + classification head -> class logits
2. Backward pass from the target CLASS LOGIT
3. For each spatial attention layer (all 27 layers):
       cam = (grad * attn).clamp(min=0).mean(heads)    [Rule 5]
       R_pp[t] += cam @ R_pp[t]                         [Rules 6+7]
   (Temporal attention runs in the forward pass but is NOT propagated into R)
4. Extract heatmap via pooling head probe (CLS-token equivalent):
       heatmap[t] = cam_cross[t] @ R_pp[t]             [Rule 10 analog]
5. Min-max normalize per frame

"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference_utils import (
    load_config, create_test_dataloader,
    setup_attribution_evaluator, match_video_ids,
    evaluate_and_print_video, print_and_save_eval_summary,
    LABEL_NAMES, LABEL_TO_IDX,
    wrap_siglip_attention_module, wrap_pooling_head_attention,
    wrap_transformer_encoder_layer, wrap_temporal_attention_module,
    chefer_attribution_renderer,
)
from visualization_video import save_lowres_visualization_video

from model.SoccerMaster.SoccerMaster_multi_task import MultiTaskingModel
from model.SoccerMaster.SoccerMaster_caption_classification import keywords_list as SOCCERMASTER_KEYWORDS


# ============================================================================
# SoccerMaster Model Config & Loading
# ============================================================================

SOCCERMASTER_DEFAULT_CONFIG = {
    'SIGLIP_BACKBONE_TYPE': 'soccer_master',
    'NUM_FRAMES': 30,
    'CKPT_PATH': 'google/siglip2-large-patch16-512',
    'TEXT_ENCODER_CKPT_PATH': 'google/siglip2-large-patch16-512',
    'BACKBONE_USE_TEMPORAL_GATE': True,
    'FREEZE_VISION_ENCODER': False,
    'FREEZE_TEXT_ENCODER': True,
    'TEMPORAL_START_LAYER': 16,
    'DATASETS_TO_HEADS': {'VideoCaption': ['CaptionClassification']},
    'BACKBONE_HIDDEN_DIM': 1024,
    'BACKBONE_TYPE': 'video',
    'CAPTION_CLASSIFICATION_DROPOUT_RATE': 0.0,
    'CAPTION_CLASSIFICATION_USE_ATTN_POOL': False,
    'CAPTION_CLASSIFICATION_USE_TRANSFORMERS': True,
    'CAPTION_CLASSIFICATION_NUM_TRANSFORMER_ENCODER': 2,
    'CAPTION_CLASSIFICATION_USE_MLP': False,
    'CAPTION_CLASSIFICATION_USE_LAYER_NORM': True,
}


def load_soccermaster_model(checkpoint_dir, device,
                            siglip2_path='google/siglip2-large-patch16-512',
                            num_frames=30):
    """Build MultiTaskingModel with CaptionClassification head and load checkpoint.

    Args:
        checkpoint_dir: Directory containing backbone.pt and CaptionClassification.pt
        device: torch device
        siglip2_path: HuggingFace model ID or local path for SigLIP2 backbone
        num_frames: Number of video frames (must match checkpoint)

    Returns:
        Loaded model on device in eval mode
    """
    config = SOCCERMASTER_DEFAULT_CONFIG.copy()
    config['CKPT_PATH'] = siglip2_path
    config['TEXT_ENCODER_CKPT_PATH'] = siglip2_path
    config['NUM_FRAMES'] = num_frames

    print(f"Building SoccerMaster model (SigLIP2: {siglip2_path}, frames: {num_frames})...")
    model = MultiTaskingModel(config)

    print(f"Loading checkpoint from: {checkpoint_dir}")
    model.load_checkpoint(checkpoint_dir)

    model = model.half().to(device).eval()
    print(f"SoccerMaster model loaded on {device} (float16)")
    return model


# ============================================================================
# SoccerMaster Attention Wrapping
# ============================================================================

def wrap_soccermaster_model(model):
    """
    Wrap SoccerMaster's attention modules for Chefer explainability.

    Wraps:
      - 27 spatial self_attn layers in VisionBackbone (R_pp accumulation).
      - Backbone temporal_attn modules in layers >= temporal_start_layer (11
        layers; per-spatial-position R_tt accumulation, Option B).
      - Pooling head cross-attention (spatial extraction).
      - 2 CaptionClassificationHead TransformerEncoder layers (broadcast
        R_tt accumulation over T frames after spatial pooling).
    """
    wrapped = {
        'spatial_attn': [],
        'temporal_attn': [],
        'pooling_head_attn': None,
        'head_te_attn': [],
    }

    vision_backbone = model.backbone.vision_model

    for i, block in enumerate(vision_backbone.encoder_blocks):
        wrapped_spatial = wrap_siglip_attention_module(block.encoder.self_attn)
        wrapped['spatial_attn'].append((i, wrapped_spatial))
        if getattr(block, 'use_temporal', False) and hasattr(block, 'temporal_attn'):
            wrapped_temporal = wrap_temporal_attention_module(block.temporal_attn)
            wrapped['temporal_attn'].append((i, wrapped_temporal))

    wrapped['pooling_head_attn'] = wrap_pooling_head_attention(vision_backbone.head.attention)

    cls_head = model.multi_task_head['CaptionClassification']
    if getattr(cls_head, 'use_transformers', False) and hasattr(cls_head, 'transformer_encoder'):
        for i, te_layer in enumerate(cls_head.transformer_encoder.layers):
            wrapped['head_te_attn'].append((i, wrap_transformer_encoder_layer(te_layer)))

    print(f"Wrapped attention modules (SoccerMaster spatial + temporal):")
    print(f"  - Backbone spatial (SigLIP2): {len(wrapped['spatial_attn'])} layers")
    print(f"  - Backbone temporal (per-position): {len(wrapped['temporal_attn'])} layers")
    print(f"  - Pooling head (probe->patches): yes")
    print(f"  - Classifier head TransformerEncoder: {len(wrapped['head_te_attn'])} layers")

    return wrapped


# ============================================================================
# Heatmap Generation
# ============================================================================

def generate_per_frame_heatmaps_soccermaster(
    model,
    video_frames: torch.Tensor,
    target_label: Optional[int] = None,
    target_label_name: Optional[str] = None,
    device: str = 'cuda',
    num_frames: int = 30,
    patch_size: int = 32,
    input_size: int = 512,
) -> np.ndarray:
    """
    Generate per-frame spatial heatmaps using Chefer method on SoccerMaster.

    Same Chefer algorithm as chefer_matchvision.py but adapted for SoccerMaster's:
    - VisionBackbone navigation (encoder_blocks instead of timesformer.resblocks)
    - Split forward pass (spatial-only layers -> temporal embedding -> temporal layers)
    - CaptionClassificationHead for logit extraction

    Args:
        model: SoccerMaster MultiTaskingModel
        video_frames: [B, C, T, H, W] video tensor
        target_label: Class index for backprop
        target_label_name: Class name string (resolved via LABEL_NAMES)
        device: Device string
        num_frames: Number of frames
        patch_size: Spatial grid size (32 for SigLIP2-large-512 = 1024 patches)
        input_size: Expected input resolution (512 for SigLIP2-large)

    Returns:
        Heatmaps [T, patch_size, patch_size] with values in [0, 1]
    """
    model.eval()
    wrapped = wrap_soccermaster_model(model)

    video_frames = video_frames.to(device)

    # Resize to SoccerMaster's expected resolution if needed
    B_orig, C_orig, T_orig, H_orig, W_orig = video_frames.shape
    if H_orig != input_size or W_orig != input_size:
        frames_flat = rearrange(video_frames, 'b c t h w -> (b t) c h w')
        frames_flat = F.interpolate(
            frames_flat, size=(input_size, input_size),
            mode='bilinear', align_corners=False,
        )
        video_frames = rearrange(frames_flat, '(b t) c h w -> b c t h w', b=B_orig, t=T_orig)

    video_frames = video_frames.half()  # float16 to match model
    video_frames.requires_grad_(True)

    B = video_frames.shape[0]
    T = num_frames
    num_patches = patch_size * patch_size  # 1024

    model.zero_grad()

    vision_backbone = model.backbone.vision_model

    # =========================================================================
    # Replicate VisionBackbone.forward() step by step.
    #
    # SoccerMaster splits its encoder into two phases:
    #   Phase 1 (layers 0 .. temporal_start_layer-1): spatial attention only
    #   Phase 2 (layers temporal_start_layer .. num_layers-1): temporal + spatial
    # Temporal embedding is injected BETWEEN the two phases.
    # =========================================================================
    x = video_frames
    B_actual, _, T_actual, _, _ = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w")

    # Patch embeddings
    x = vision_backbone.vision_model_embedding(x)  # [B*T, 1024, 1024]

    # Phase 1: spatial-only layers (before temporal_start_layer)
    for idx in range(vision_backbone.temporal_start_layer):
        x = vision_backbone.encoder_blocks[idx](x, B_actual, T_actual)

    # Add temporal embedding (injected between spatial and temporal blocks)
    x = rearrange(x, '(b t) n m -> b n t m', b=B_actual, t=T_actual)
    x = x + vision_backbone.temporal_embedding
    x = rearrange(x, 'b n t m -> (b t) n m')

    # Phase 2: temporal + spatial layers
    for idx in range(vision_backbone.temporal_start_layer, vision_backbone.num_layers):
        x = vision_backbone.encoder_blocks[idx](x, B_actual, T_actual)

    # Post-norm and pooling head
    x = vision_backbone.post_norm(x)
    x = vision_backbone.head(x)  # [B*T, 1024]
    video_features = rearrange(x, '(b t) m -> b t m', b=B_actual, t=T_actual)

    # Convert label name to index if provided
    if target_label_name is not None:
        if target_label_name.lower() not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{target_label_name}'. Available: {LABEL_NAMES}")
        target_label = LABEL_TO_IDX[target_label_name.lower()]
        print(f"Resolved label name '{target_label_name}' to index {target_label}")

    # =========================================================================
    # Route through CaptionClassificationHead for logit extraction.
    # =========================================================================
    backbone_outputs = {'global_features': video_features}
    cls_head = model.multi_task_head['CaptionClassification']
    cls_output = cls_head(backbone_outputs, metas=None)
    cls_logits = cls_output['logits']  # [B, 23]

    # Backprop from target class logit
    target = cls_logits[0, target_label]
    print(f"Chefer class-logit backprop: class_idx={target_label}, logit={target.item():.4f}")

    # CRITICAL: Backward pass to compute gradients through attention
    target.backward(retain_graph=True)

    print(f"Computed backward pass. Target value: {target.item():.4f}")

    # =========================================================================
    # Four-matrix relevance propagation (see chefer_matchvision_fourmatrix.py
    # for the full derivation; this is the SoccerMaster-specific application).
    #
    # State:
    #   R_T [N, T, T]    fp32  per-spatial-position temporal relevance, init=I
    #   R_S [T, N, N]    fp32  per-frame spatial relevance, init=I
    #   R_X [T, N, T, N] fp16  cross-axis joint relevance, init=0
    # SoccerMaster: N=1024, T=30. R_X = 1.9 GB fp16.
    #
    # Per-block updates (interleaved, in forward order):
    #   Layers 0-15: spatial sub-block only (use_temporal=False)
    #   Layers 16-23: temporal sub-block (gated g=tanh(α)) then spatial sub-block
    # After backbone: 2 head TE layers contribute to R_T (broadcast across N).
    # =========================================================================

    spatial_by_layer = {layer_idx: m for (layer_idx, m) in wrapped['spatial_attn']}
    temporal_by_layer = {layer_idx: m for (layer_idx, m) in wrapped['temporal_attn']}
    num_backbone_layers = max(spatial_by_layer.keys()) + 1 if spatial_by_layer else 0

    R_T = torch.eye(T, device=device, dtype=torch.float32)\
        .unsqueeze(0).expand(num_patches, T, T).contiguous()              # [N, T, T]
    R_S = torch.eye(num_patches, device=device, dtype=torch.float32)\
        .unsqueeze(0).expand(T, num_patches, num_patches).contiguous()    # [T, N, N]
    R_X = torch.zeros(T, num_patches, T, num_patches,
                      device=device, dtype=torch.float16)                  # [T, N, T, N]

    encoder_blocks = vision_backbone.encoder_blocks
    per_layer_times = []

    for layer_idx in range(num_backbone_layers):
        layer_t0 = time.time()

        # ----- Temporal sub-block (gated; only layers with use_temporal=True) -----
        if layer_idx in temporal_by_layer:
            t_attn_module = temporal_by_layer[layer_idx]
            attn = t_attn_module.get_attn()
            grad = t_attn_module.get_attn_gradients()
            if attn is not None and grad is not None:
                A_T = (grad.float() * attn.float()).clamp(min=0).mean(dim=1)  # [N, T, T]
                g = float(encoder_blocks[layer_idx].temporal_alpha_attn.tanh().item())

                if g != 0.0:
                    R_X_f = R_X.float()
                    R_X_diag_t = R_X_f.diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()  # [T, N, N]
                    A_T_diag = A_T.diagonal(dim1=1, dim2=2)                                     # [N, T]

                    delta_R_T_temp = g * torch.bmm(A_T, R_T)                                    # [N, T, T]

                    delta_R_S_self = g * A_T_diag.T.unsqueeze(-1) * R_S                         # [T, N, N]
                    full_S = torch.einsum('abc, cabd -> bad', A_T, R_X_f)                       # [T, N, N]
                    sub_S = A_T_diag.T.unsqueeze(-1) * R_X_diag_t                               # [T, N, N]
                    delta_R_S_cross = g * (full_S - sub_S)
                    del full_S, sub_S

                    delta_R_X_temp = torch.einsum('abc, caed -> baed', A_T, R_X_f)              # [T, N, T, N]
                    delta_R_X_temp.sub_(torch.einsum('abe, ead -> baed', A_T, R_X_diag_t))
                    delta_R_X_temp.add_(torch.einsum('abe, ead -> baed', A_T, R_S))
                    delta_R_X_temp.mul_(g)

                    R_T = R_T + delta_R_T_temp
                    R_S = R_S + delta_R_S_self + delta_R_S_cross
                    R_X_f.add_(delta_R_X_temp)
                    del delta_R_X_temp, R_X_diag_t, delta_R_T_temp
                    del delta_R_S_self, delta_R_S_cross
                    R_X = R_X_f.half()
                    del R_X_f

        # ----- Spatial sub-block (no gate) -----
        if layer_idx in spatial_by_layer:
            s_attn_module = spatial_by_layer[layer_idx]
            attn = s_attn_module.get_attn()
            grad = s_attn_module.get_attn_gradients()
            if attn is not None and grad is not None:
                A_S = (grad.float() * attn.float()).clamp(min=0).mean(dim=1)  # [T, N, N]
                A_S_diag = A_S.diagonal(dim1=1, dim2=2)                       # [T, N]

                R_X_f = R_X.float()
                R_X_diag_p = R_X_f.diagonal(dim1=1, dim2=3).contiguous()      # [T, T, N]

                delta_R_S_spat = torch.bmm(A_S, R_S)                          # [T, N, N]

                delta_R_T_self = A_S_diag.T.unsqueeze(-1) * R_T               # [N, T, T]
                full_T = torch.einsum('bac, bcda -> abd', A_S, R_X_f)         # [N, T, T]
                sub_T = torch.einsum('ba, bda -> abd', A_S_diag, R_X_diag_p)  # [N, T, T]
                delta_R_T_cross = full_T - sub_T
                del full_T, sub_T

                # Spatial sub-block: diffusion only on R_X (no injection).
                full_X_spat = torch.einsum('bac, badc -> bad', A_S, R_X_f)    # [T, N, T]
                delta_R_X_spat = full_X_spat.unsqueeze(-1) - A_S.unsqueeze(2) * R_X_f  # [T, N, T, N]
                del full_X_spat

                R_S = R_S + delta_R_S_spat
                R_T = R_T + delta_R_T_self + delta_R_T_cross
                R_X_f.add_(delta_R_X_spat)
                del delta_R_X_spat, R_X_diag_p
                del delta_R_S_spat, delta_R_T_self, delta_R_T_cross
                R_X = R_X_f.half()
                del R_X_f

        per_layer_times.append(time.time() - layer_t0)

    # ----- Head TE layers (after backbone) — update R_T only, broadcast over N -----
    for layer_idx, te_attn in wrapped['head_te_attn']:
        cam_h = te_attn.get_attn()
        grad_h = te_attn.get_attn_gradients()
        if cam_h is None or grad_h is None:
            print(f'  WARNING: head TE layer {layer_idx} attn/grad not captured.')
            continue
        cam_h = cam_h.reshape(-1, T, T).float()
        grad_h = grad_h.reshape(-1, T, T).float()
        cam_global = (grad_h * cam_h).clamp(min=0).mean(dim=0)               # [T, T]
        R_T = R_T + cam_global @ R_T

    print(f'  Per-backbone-layer four-matrix update times (sec): '
          f'mean={np.mean(per_layer_times):.4f}  max={max(per_layer_times):.4f}  '
          f'total={sum(per_layer_times):.3f}')

    # =========================================================================
    # Pooling-head cam_pool[t], same as chefer_*_temporal.py.
    # =========================================================================
    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()           # [B*T, num_heads, 1, N]
    head_grad = head_attn_module.get_attn_gradients()
    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")

    cam_pool = torch.zeros(T, num_patches, device=device, dtype=torch.float32)
    for t in range(T):
        cam = head_attn[t]
        grad = head_grad[t]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = (grad * cam).clamp(min=0).mean(dim=0)
        cam_pool[t] = cam.float().squeeze(0)

    H_self = torch.einsum('tn, tnp -> tp', cam_pool, R_S)                     # [T, N]
    H_cross = torch.einsum('ab, abcd -> cd', cam_pool, R_X.float())           # [T, N]
    image_relevance = H_self + H_cross                                         # [T, N]

    heatmaps = []
    for t in range(T):
        hm = image_relevance[t].detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(hm)
    heatmaps = np.stack(heatmaps, axis=0).astype(np.float32)  # [T, patch_size, patch_size]

    # Per-frame temporal weights from R_T (same as temporal pipeline).
    w = R_T.mean(dim=(0, 1)).detach().cpu().numpy()
    temporal_weights = np.ones(T, dtype=np.float32)
    if w.max() > 0:
        temporal_weights = (w / w.max()).astype(np.float32)
    print(f"  R_T temporal weights (normalized): "
          f"min={temporal_weights.min():.3f} max={temporal_weights.max():.3f} "
          f"argmax_frame={int(temporal_weights.argmax())}")

    # Min-max normalization per frame, then scale by temporal weight so that
    # cross-frame mean intensity (the signal T-IoU consumes) reflects the
    # head's temporal attention rather than uniform [0,1].
    for t in range(T):
        h = heatmaps[t]
        if h.max() > h.min():
            heatmaps[t] = (h - h.min()) / (h.max() - h.min())
        else:
            heatmaps[t] = np.zeros_like(h)
        heatmaps[t] *= temporal_weights[t]

    return heatmaps


# ============================================================================
# Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chefer explainability inference (per-frame spatial-only) for SoccerMaster classification.')
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_ibex.py',
                        help='Path to the dataset Python config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to SoccerMaster checkpoint dir (backbone.pt + CaptionClassification.pt)')
    parser.add_argument('--siglip2_path', type=str,
                        default='google/siglip2-large-patch16-512',
                        help='HuggingFace model ID or local path for SigLIP2 backbone')
    parser.add_argument('--coco_json', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'annotations-coco.json'),
                        help='Path to annotations-coco.json for attribution evaluation')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default=None,
                        help='Optional path to save evaluation results as JSON')
    parser.add_argument('--output_dir', type=str,
                        default='../output_chefer_soccermaster/',
                        help='Directory to save attribution visualization outputs')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input resolution for SoccerMaster (default: 512)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load dataset config
    # ----------------------------------------------------------------
    config = load_config(args.config_path)

    config_dataset = config['dataset']
    config_test_dataset = config_dataset['test']

    config_training_settings = config['training_settings']
    device_ids = config_training_settings['device_ids']
    device = torch.device(f'cuda:{device_ids[0]}')

    # ----------------------------------------------------------------
    # Dataset (reuse existing dataloader)
    # ----------------------------------------------------------------
    test_dataset, test_data_loader = create_test_dataloader(config_test_dataset)

    # ----------------------------------------------------------------
    # SoccerMaster Model
    # ----------------------------------------------------------------
    model = load_soccermaster_model(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        siglip2_path=args.siglip2_path,
        num_frames=30,
    )

    print(f"SoccerMaster model loaded. Keywords ({len(SOCCERMASTER_KEYWORDS)}): {SOCCERMASTER_KEYWORDS}")

    # Compute patch_size from input_size
    patch_size = args.input_size // 16  # 512 / 16 = 32
    print(f"Input size: {args.input_size}, patch grid: {patch_size}x{patch_size} = {patch_size**2} patches")

    # ----------------------------------------------------------------
    # COCO evaluator (optional)
    # ----------------------------------------------------------------
    coco_json_path = os.path.abspath(args.coco_json)
    attribution_evaluator, all_eval_results = setup_attribution_evaluator(coco_json_path)

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Chefer SoccerMaster Inference')

    for _, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:
        video_name = video_path[0].split('/')[-1]
        print(f'\nProcessing video: {video_name}')

        matched_video_ids = match_video_ids(attribution_evaluator, video_path[0])

        frames = frames.to(device)
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)

        # ----------------------------------------------------------
        # Run Chefer explainability
        # ----------------------------------------------------------
        target_label_idx = caption[0].item()

        chefer_heatmaps = generate_per_frame_heatmaps_soccermaster(
            model,
            frames,
            target_label=target_label_idx,
            device=str(device),
            num_frames=frames.shape[2],
            patch_size=patch_size,
            input_size=args.input_size,
        )
        # chefer_heatmaps: [T, 32, 32] numpy float32 in [0, 1]
        print(f'Chefer SoccerMaster heatmap shape: {chefer_heatmaps.shape}')

        # Compute per-frame attribution scores (mean heatmap intensity)
        chefer_scores = chefer_heatmaps.mean(axis=(1, 2))  # [T]
        print(f'Chefer scores shape: {chefer_scores.shape}')

        # ----------------------------------------------------------
        # Get model predictions (fresh forward pass without hooks)
        # ----------------------------------------------------------
        with torch.no_grad():
            # Resize for SoccerMaster and rearrange to [B, T, C, H, W]
            B_f, C_f, T_f, H_f, W_f = frames.shape
            frames_flat = rearrange(frames, 'b c t h w -> (b t) c h w')
            if H_f != args.input_size or W_f != args.input_size:
                frames_flat = F.interpolate(
                    frames_flat, size=(args.input_size, args.input_size),
                    mode='bilinear', align_corners=False,
                )
            frames_for_pred = rearrange(frames_flat, '(b t) c h w -> b t c h w', b=B_f, t=T_f)
            frames_for_pred = frames_for_pred.half()  # float16 to match model

            backbone_out = model.backbone(frames_for_pred)
            cls_out = model.multi_task_head['CaptionClassification'](backbone_out, metas=None)
            logits = cls_out['logits']  # [B, 23]

        i = 0
        for sample_idx in range(frames.shape[0]):
            new_frames = dummy_frames[sample_idx]
            new_frames = new_frames.permute(0, 2, 3, 1)  # [T, H, W, C]

            # ----------------------------------------------------------
            # COCO evaluation
            # ----------------------------------------------------------
            evaluate_and_print_video(
                attribution_evaluator, chefer_heatmaps, matched_video_ids,
                video_name, args.cam_threshold, all_eval_results,
            )

            # ----------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------
            _, predictions = torch.topk(logits, k=5, dim=1, largest=True, sorted=True)
            prediction_text = test_dataset.keywords[predictions[sample_idx, 0].item()]
            ground_truth_text = caption_text[sample_idx]
            print(f'Prediction: {prediction_text}')
            print(f'Ground Truth: {ground_truth_text}')

            # ----------------------------------------------------------
            # Visualization
            # ----------------------------------------------------------
            save_lowres_visualization_video(
                video_directory=video_directory,
                video_name=video_timestamp,
                lowres_frames=new_frames,
                attribution_maps=chefer_heatmaps,
                attribution_scores=chefer_scores,
                prediction_text=prediction_text,
                ground_truth_text=ground_truth_text,
                attribution_evaluator=attribution_evaluator,
                matched_video_id=matched_video_ids[0] if attribution_evaluator and matched_video_ids else None,
                cam_threshold=args.cam_threshold,
                attribution_method_name='Chefer-SoccerMaster',
                attribution_renderer=chefer_attribution_renderer,
            )

            i += 1

        all_predictions.append(predictions.cpu())
        del frames

    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    # ----------------------------------------------------------------
    # Global evaluation summary
    # ----------------------------------------------------------------
    if attribution_evaluator is not None and all_eval_results:
        eval_output = args.eval_output_json
        if eval_output is None:
            eval_output = os.path.join(args.output_dir, 'chefer_soccermaster_eval_results.json')
        print_and_save_eval_summary(
            all_eval_results,
            eval_output_path=eval_output,
            summary_title='Chefer SoccerMaster Attribution Label-Group Evaluation Summary',
        )


if __name__ == '__main__':
    main()
