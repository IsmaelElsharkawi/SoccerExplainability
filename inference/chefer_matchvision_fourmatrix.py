"""
Chefer explainability inference for MatchVision classification

Each frame has an independent R_pp [N, N] relevance matrix that only accumulates spatial attention.

================================================================================
CHEFER METHOD (per-frame spatial-only)
================================================================================

Based on:
    "Generic Attention-model Explainability for Interpreting Bi-Modal and
     Encoder-Decoder Transformers" (Chefer et al., ICCV 2021)

1. Forward pass through VisionTimesformer + classification head
2. Backward pass from the target CLASS LOGIT
3. For each spatial attention layer:
       cam = (grad * attn).clamp(min=0).mean(heads)    [Rule 5]
       R_pp[t] += cam @ R_pp[t]                         [Rules 6+7]
   (Temporal attention is NOT propagated into R)
4. Extract heatmap via pooling head probe (CLS-token equivalent):
       heatmap[t] = cam_cross[t] @ R_pp[t]             [Rule 10 analog]
5. Min-max normalize per frame

================================================================================
ARCHITECTURE ADAPTATION
================================================================================

- SigLIP's vision encoder has no CLS token (unlike ViT-B/16).  Its pooling
  head uses a learnable probe that cross-attends to all 196 patches, so
  cam_cross @ R_pp is the direct analog of R[cls, 1:] in original Chefer implementation.
- Temporal attention (Timesformer) is present in the forward pass but is
  NOT wrapped and NOT included in the relevance propagation.
- All spatial layers are used (no layer skipping).

================================================================================
LABEL MAPPING (LABEL_NAMES, alphabetical)
================================================================================

Use target_label / target_label_name to specify class:

    0:  var                  8:  shot off target     16: goal
    1:  end of half game     9:  start of half game  17: penalty
    2:  clearance           10:  substitution        18: yellow card
    3:  second yellow card  11:  saved by goal-keeper 19: foul lead to penalty
    4:  injury              12:  red card            20: corner
    5:  ball possession     13:  lead to corner      21: free kick
    6:  throw in            14:  ball out of play    22: foul with no card
    7:  show added time     15:  off side

Usage:
    python chefer_matchvision.py \\
        --config_path ../config/pretrain_classification_ibex.py \\
        --checkpoint_path /path/to/pretrained_classification.pth \\
        --output_dir /path/to/output
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference_utils import (
    load_config, create_test_dataloader, load_classifier,
    setup_attribution_evaluator, match_video_ids,
    evaluate_and_print_video, print_and_save_eval_summary,
    LABEL_NAMES, LABEL_TO_IDX,
    wrap_siglip_attention_module, wrap_pooling_head_attention,
    wrap_transformer_encoder_layer, wrap_temporal_attention_module,
    chefer_attribution_renderer,
)
from visualization_video import save_lowres_visualization_video


# ============================================================================
# Model Wrapping
# ============================================================================

def wrap_matchvision_model(model):
    """
    Wrap MatchVision_Classifier's attention modules for Chefer explainability.

    Wraps:
      - 12 spatial self_attn layers in VisionTimesformer (R_pp accumulation).
      - 12 backbone temporal_attn modules (per-spatial-position R_tt accumulation).
      - Pooling head cross-attention (spatial extraction).
      - 2 classifier head TransformerEncoder layers (broadcast R_tt accumulation).

    This is the spatial+temporal Chefer pipeline (Option B): backbone temporal
    relevance is tracked per spatial position via R_tt[N, T, T], then collapsed
    to per-frame weights only at the final extraction step.
    """
    wrapped = {
        'spatial_attn': [],
        'temporal_attn': [],
        'pooling_head_attn': None,
        'head_te_attn': [],
    }

    visual_encoder = model.siglip_model
    timesformer = visual_encoder.timesformer

    for i, block in enumerate(timesformer.resblocks):
        wrapped_spatial = wrap_siglip_attention_module(block.encoder.self_attn)
        wrapped['spatial_attn'].append((i, wrapped_spatial))
        if hasattr(block, 'temporal_attn'):
            wrapped_temporal = wrap_temporal_attention_module(block.temporal_attn)
            wrapped['temporal_attn'].append((i, wrapped_temporal))

    wrapped['pooling_head_attn'] = wrap_pooling_head_attention(visual_encoder.head.attention)

    if getattr(model, 'use_transformer', False) and hasattr(model, 'transformer_encoder'):
        for i, te_layer in enumerate(model.transformer_encoder.layers):
            wrapped['head_te_attn'].append((i, wrap_transformer_encoder_layer(te_layer)))

    print(f"Wrapped attention modules (spatial + temporal):")
    print(f"  - Backbone spatial (SigLIP): {len(wrapped['spatial_attn'])} layers")
    print(f"  - Backbone temporal (per-position): {len(wrapped['temporal_attn'])} layers")
    print(f"  - Pooling head (probe->patches): yes")
    print(f"  - Classifier head TransformerEncoder: {len(wrapped['head_te_attn'])} layers")

    return wrapped


# ============================================================================
# Heatmap Generation
# ============================================================================

def generate_per_frame_heatmaps(
    model,
    video_frames: torch.Tensor,
    target_label: Optional[int] = None,
    target_label_name: Optional[str] = None,
    device: str = 'cuda',
    num_frames: int = 30,
    patch_size: int = 14,  # 14x14 = 196 patches for 224x224 input
) -> np.ndarray:
    """
    Generate per-frame spatial heatmaps using the Chefer method (ICCV 2021).
    
    Per-frame spatial-only version: each frame has an independent R_pp [N, N]
    relevance matrix. Only spatial attention layers propagate relevance.
    Temporal attention is present in the forward pass but excluded from R.
    
    Algorithm (matches example.py exactly, applied per-frame):
        1. Forward pass -> class logits
        2. Backward from target class logit
        3. For each spatial attention layer:
               cam = (grad * attn).clamp(min=0).mean(heads)
               R_pp[t] += cam @ R_pp[t]
        4. Extract via pooling head probe: heatmap = cam_cross @ R_pp[t]
           (analog of R[cls, 1:] for CLS-less SigLIP)
        5. Min-max normalize per frame
    
    Args:
        model: MatchVision model with model.classifier for class logits
        video_frames: Video tensor [B, C, T, H, W]
        target_label: Class index for backprop (use with target_label_name)
        target_label_name: Class name string (resolved via LABEL_NAMES)
        device: Device to run on
        num_frames: Number of frames (T dimension)
        patch_size: Spatial grid size (14 for SigLIP-base = 196 patches)
    
    Returns:
        Heatmaps array [T, patch_size, patch_size] with values in [0, 1]
    """
    model.eval()
    wrapped = wrap_matchvision_model(model)

    video_frames = video_frames.to(device)
    video_frames.requires_grad_(True)

    B = video_frames.shape[0]
    T = num_frames
    num_patches = patch_size * patch_size  # 196

    model.zero_grad()

    # =========================================================================
    # Forward pass through VisionTimesformer
    # =========================================================================
    visual_encoder = model.siglip_model

    x = video_frames
    B_actual, _, T_actual, _, _ = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w")

    x = visual_encoder.vision_model_embedding(x)  # [B*T, 196, 768]
    x = rearrange(x, "(b t) n m -> b n t m", b=B_actual, t=T_actual)
    x = x + visual_encoder.temporal_positional_embedding
    x = rearrange(x, "b n t m -> (b t) n m")

    x = visual_encoder.timesformer(x, B_actual, T_actual)

    x = visual_encoder.post_layernorm(x)
    x = visual_encoder.head(x)  # [B*T, 768]
    video_features = rearrange(x, "(b t) m -> b t m", b=B_actual, t=T_actual)

    # Convert label name to index if provided
    if target_label_name is not None:
        if target_label_name.lower() not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{target_label_name}'. Available: {LABEL_NAMES}")
        target_label = LABEL_TO_IDX[target_label_name.lower()]
        print(f"Resolved label name '{target_label_name}' to index {target_label}")

    # =========================================================================
    # Classification head: LN -> TransformerEncoder(avg_pool) -> LN -> Linear
    # =========================================================================
    x_cls = model.classifier_ln1(video_features)  # [B, T, 768]
    x_cls = x_cls.permute(1, 0, 2)                # [T, B, 768]
    x_cls = model.transformer_encoder(x_cls)
    x_cls = x_cls.mean(dim=0)                      # [B, 768]
    x_cls = model.classifier_ln2(x_cls)
    cls_logits = model.classifier(x_cls)            # [B, num_classes]
    
    # Backprop from the target class logit (one-hot selection)
    target = cls_logits[0, target_label]
    print(f"Chefer class-logit backprop: class_idx={target_label}, logit={target.item():.4f}")
    
    # CRITICAL: Backward pass to compute gradients through attention
    target.backward(retain_graph=True)
    
    print(f"Computed backward pass. Target value: {target.item():.4f}")
    
    # =========================================================================
    # Relevance propagation — mirrors example.py lines 23-30 exactly.
    # Only addition: per-frame loop (temporal axis from Timesformer).
    #
    # Original (example.py):
    #   for blk in image_attn_blocks:
    #       grad = blk.attn_grad
    #       cam = blk.attn_probs
    #       cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    #       grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    #       cam = grad * cam
    #       cam = cam.clamp(min=0).mean(dim=0)
    #       R += torch.matmul(cam, R)
    # =========================================================================
    # =========================================================================
    # Four-matrix relevance propagation
    #
    # State (for the duration of this function):
    #   R_T [N, T, T]     per-spatial-position temporal relevance, init=I
    #                     indexed as R_T[p][t_query, t_source]
    #   R_S [T, N, N]     per-frame spatial relevance, init=I
    #                     indexed as R_S[t][p_query, p_source]
    #   R_X [T, N, T, N]  cross-axis joint relevance, init=0, stored fp16
    #                     indexed as R_X[t_query, p_query, t_source, p_source]
    #
    # Per backbone layer, in forward order:
    #   (a) Temporal sub-block (gated g = tanh(alpha_layer)):
    #         Updates R_T (per p), R_S (self+cross from R_X), R_X (diffusion+inject from R_S)
    #   (b) Spatial sub-block (no gate):
    #         Updates R_S (per t), R_T (self+cross from R_X), R_X (diffusion only)
    # After all backbone layers, head TE layers contribute to R_T (broadcast across N).
    #
    # Extraction:
    #   cam_pool[t] from pooling head (per frame, [N])
    #   H_self[t, p]  = cam_pool[t] @ R_S[t]                          # [T, N]
    #   H_cross[t, p] = sum_{t', p_q} cam_pool[t', p_q] · R_X[t', p_q, t, p]
    #   image_relevance = H_self + H_cross
    # Then per-frame temporal weight w[t] from R_T (mean over spatial positions
    # and output rows, max-normalized) is applied multiplicatively per frame.
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

    resblocks = model.siglip_model.timesformer.resblocks
    per_layer_times = []

    for layer_idx in range(num_backbone_layers):
        layer_t0 = time.time()

        # ----- Temporal sub-block (gated by g = tanh(alpha_layer)) -----
        if layer_idx in temporal_by_layer:
            t_attn_module = temporal_by_layer[layer_idx]
            attn = t_attn_module.get_attn()
            grad = t_attn_module.get_attn_gradients()
            if attn is not None and grad is not None:
                A_T = (grad.float() * attn.float()).clamp(min=0).mean(dim=1)  # [N, T, T]
                g = float(resblocks[layer_idx].temporal_alpha_attn.tanh().item())

                if g != 0.0:
                    R_X_f = R_X.float()
                    R_X_diag_t = R_X_f.diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()  # [T, N, N]: R_X[t, a, t, d]
                    A_T_diag = A_T.diagonal(dim1=1, dim2=2)                                     # [N, T]: A_T[a, b, b]

                    # delta R_T = g · A_T @ R_T  (per-position bmm)
                    delta_R_T_temp = g * torch.bmm(A_T, R_T)                                    # [N, T, T]

                    # delta R_S: g · ( A_T_diag · R_S  +  Σ_{t'≠t} A_T[p, t, t'] · R_X[t', p, t, p'] )
                    delta_R_S_self = g * A_T_diag.T.unsqueeze(-1) * R_S                         # [T, N, N]
                    full_S = torch.einsum('abc, cabd -> bad', A_T, R_X_f)                       # [T, N, N]
                    sub_S = A_T_diag.T.unsqueeze(-1) * R_X_diag_t                               # [T, N, N]
                    delta_R_S_cross = g * (full_S - sub_S)
                    del full_S, sub_S

                    # delta R_X: g · ( diffusion(over t''≠t') + injection(R_S) )
                    delta_R_X_temp = torch.einsum('abc, caed -> baed', A_T, R_X_f)              # [T, N, T, N] full
                    delta_R_X_temp.sub_(torch.einsum('abe, ead -> baed', A_T, R_X_diag_t))      # subtract t''=t'
                    delta_R_X_temp.add_(torch.einsum('abe, ead -> baed', A_T, R_S))             # injection
                    delta_R_X_temp.mul_(g)

                    R_T = R_T + delta_R_T_temp
                    R_S = R_S + delta_R_S_self + delta_R_S_cross
                    R_X = (R_X_f + delta_R_X_temp).half()
                    del R_X_f, R_X_diag_t, delta_R_X_temp, delta_R_T_temp
                    del delta_R_S_self, delta_R_S_cross
                # If g == 0.0, the temporal sub-block contributes nothing — skip.

        # ----- Spatial sub-block (no gate; standard SigLIP encoder layer) -----
        if layer_idx in spatial_by_layer:
            s_attn_module = spatial_by_layer[layer_idx]
            attn = s_attn_module.get_attn()
            grad = s_attn_module.get_attn_gradients()
            if attn is not None and grad is not None:
                A_S = (grad.float() * attn.float()).clamp(min=0).mean(dim=1)  # [T, N, N]
                A_S_diag = A_S.diagonal(dim1=1, dim2=2)                       # [T, N]

                R_X_f = R_X.float()
                R_X_diag_p = R_X_f.diagonal(dim1=1, dim2=3).contiguous()      # [T, T, N]: R_X[b, a, d, a]

                # delta R_S = A_S @ R_S  (per-frame bmm)
                delta_R_S_spat = torch.bmm(A_S, R_S)                          # [T, N, N]

                # delta R_T: A_S_diag · R_T  +  Σ_{p'≠p} A_S[t, p, p'] · R_X[t, p', t', p]
                delta_R_T_self = A_S_diag.T.unsqueeze(-1) * R_T               # [N, T, T]
                full_T = torch.einsum('bac, bcda -> abd', A_S, R_X_f)         # [N, T, T]
                sub_T = torch.einsum('ba, bda -> abd', A_S_diag, R_X_diag_p)  # [N, T, T]
                delta_R_T_cross = full_T - sub_T
                del full_T, sub_T

                # delta R_X: diffusion only (no injection per architectural asymmetry —
                # spatial sub-block has no gated cross-axis residual, only the temporal
                # sub-block's tanh(α)-gated residual creates cross-axis relevance.)
                #   diffusion = full_sum - subtract(c=e)
                #   full: 'bac, badc -> bad' (sum over c=p''), broadcast over e
                #   sub:  A_S[b, a, e] · R_X[b, a, d, e]  elementwise
                full_X_spat = torch.einsum('bac, badc -> bad', A_S, R_X_f)    # [T, N, T]
                delta_R_X_spat = full_X_spat.unsqueeze(-1) - A_S.unsqueeze(2) * R_X_f  # [T, N, T, N]
                del full_X_spat

                R_S = R_S + delta_R_S_spat
                R_T = R_T + delta_R_T_self + delta_R_T_cross
                R_X = (R_X_f + delta_R_X_spat).half()
                del R_X_f, R_X_diag_p, delta_R_X_spat
                del delta_R_S_spat, delta_R_T_self, delta_R_T_cross

        per_layer_times.append(time.time() - layer_t0)

    # ----- Head TE layers (after backbone) -----
    # Single-axis temporal self-attention; updates R_T only (broadcast across N).
    # Does not touch R_S or R_X (no spatial axis at this point).
    for layer_idx, te_attn in wrapped['head_te_attn']:
        cam_h = te_attn.get_attn()
        grad_h = te_attn.get_attn_gradients()
        if cam_h is None or grad_h is None:
            print(f'  WARNING: head TE layer {layer_idx} attn/grad not captured.')
            continue
        cam_h = cam_h.reshape(-1, T, T).float()
        grad_h = grad_h.reshape(-1, T, T).float()
        cam_global = (grad_h * cam_h).clamp(min=0).mean(dim=0)               # [T, T]
        R_T = R_T + cam_global @ R_T                                          # broadcast over N

    print(f'  Per-backbone-layer four-matrix update times (sec): '
          f'mean={np.mean(per_layer_times):.4f}  max={max(per_layer_times):.4f}  '
          f'total={sum(per_layer_times):.3f}')

    # =========================================================================
    # Pooling-head extraction: cam_pool[t] for each frame (gradient-weighted
    # probe-to-patches attention).
    # =========================================================================
    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()           # [B*T, num_heads, 1, N]
    head_grad = head_attn_module.get_attn_gradients()  # [B*T, num_heads, 1, N]
    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")

    cam_pool = torch.zeros(T, num_patches, device=device, dtype=torch.float32)
    for t in range(T):
        cam = head_attn[t]
        grad = head_grad[t]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = (grad * cam).clamp(min=0).mean(dim=0)  # [1, N]
        cam_pool[t] = cam.float().squeeze(0)         # [N]

    # H_self[t, p] = cam_pool[t] @ R_S[t][:, p]
    H_self = torch.einsum('tn, tnp -> tp', cam_pool, R_S)                     # [T, N]
    # H_cross[t, p] = sum_{t', p_q} cam_pool[t', p_q] · R_X[t', p_q, t, p]
    H_cross = torch.einsum('ab, abcd -> cd', cam_pool, R_X.float())           # [T, N]
    image_relevance = H_self + H_cross                                         # [T, N]

    heatmaps = []
    for t in range(T):
        hm = image_relevance[t].detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(hm)
    heatmaps = np.stack(heatmaps, axis=0)  # [T, patch_size, patch_size]

    # =========================================================================
    # Per-frame temporal weights from R_T (same as chefer_*_temporal.py).
    # =========================================================================
    w = R_T.mean(dim=(0, 1)).detach().cpu().numpy()                           # [T]
    temporal_weights = np.ones(T, dtype=np.float32)
    if w.max() > 0:
        temporal_weights = (w / w.max()).astype(np.float32)
    print(f"  R_T temporal weights (normalized): "
          f"min={temporal_weights.min():.3f} max={temporal_weights.max():.3f} "
          f"argmax_frame={int(temporal_weights.argmax())}")

    # Per-frame min-max normalize, multiply by temporal weight.
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
        description='Chefer explainability inference (per-frame spatial-only) for MatchVision classification.')
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_ibex.py',
                        help='Path to the Python config file')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/path/to/pretrained_classification.pth',
                        help='Path to the checkpoint file')
    parser.add_argument('--coco_json', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'annotations-coco.json'),
                        help='Path to annotations-coco.json for attribution evaluation')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default=None,
                        help='Optional path to save per-video evaluation results as JSON')
    parser.add_argument('--output_dir', type=str,
                        default='../output_chefer_soccer/',
                        help='Directory to save attribution visualization outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load config and setup
    # ----------------------------------------------------------------
    config = load_config(args.config_path)
    checkpoint_path = args.checkpoint_path

    config_dataset = config['dataset']
    config_test_dataset = config_dataset['test']

    config_training_settings = config['training_settings']
    device_ids = config_training_settings['device_ids']
    classifier_transformer_type = config_training_settings['classifier_transformer_type']
    encoder_type = config_training_settings['encoder_type']
    use_transformer = config_training_settings['use_transformer']

    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    test_dataset, test_data_loader = create_test_dataloader(config_test_dataset)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    classifier = load_classifier(
        config_test_dataset, classifier_transformer_type, encoder_type,
        use_transformer, checkpoint_path, devices, device_ids,
    )

    print(f"Model loaded. Keywords ({len(test_dataset.keywords)}): {test_dataset.keywords}")

    # ----------------------------------------------------------------
    # COCO evaluator (optional)
    # ----------------------------------------------------------------
    coco_json_path = os.path.abspath(args.coco_json)
    attribution_evaluator, all_eval_results = setup_attribution_evaluator(coco_json_path)

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Chefer Per-Frame Inference')

    for _, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:
        video_name = video_path[0].split('/')[-1]
        print(f'\nProcessing video: {video_name}')

        matched_video_ids = match_video_ids(attribution_evaluator, video_path[0])

        frames = frames.to(devices[0])
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)

        # ----------------------------------------------------------
        # Run Chefer explainability
        # ----------------------------------------------------------
        target_label_idx = caption[0].item()

        chefer_heatmaps = generate_per_frame_heatmaps(
            classifier.module,
            frames,
            target_label=target_label_idx,
            device=str(devices[0]),
            num_frames=frames.shape[2],
        )
        # chefer_heatmaps: [T, 14, 14] numpy float32 in [0, 1]
        print(f'Chefer per-frame heatmap shape: {chefer_heatmaps.shape}')

        # Compute per-frame attribution scores (mean heatmap intensity)
        chefer_scores = chefer_heatmaps.mean(axis=(1, 2))  # [T]
        print(f'Chefer scores shape: {chefer_scores.shape}')

        # ----------------------------------------------------------
        # Get model predictions (fresh forward pass without hooks)
        # ----------------------------------------------------------
        with torch.no_grad():
            logits = classifier.module.forward(frames)

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
            predictions = classifier.module.get_types(logits)
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
                attribution_method_name='Chefer-PerFrame',
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
            eval_output = os.path.join(args.output_dir, 'chefer_eval_results.json')
        print_and_save_eval_summary(
            all_eval_results,
            eval_output_path=eval_output,
            summary_title='Chefer Per-Frame Attribution Label-Group Evaluation Summary',
        )


if __name__ == '__main__':
    main()
