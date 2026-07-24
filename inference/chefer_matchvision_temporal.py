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
        --config_path ../config/pretrain_classification_cluster.py \\
        --checkpoint_path /path/to/pretrained_classification.pth \\
        --output_dir /path/to/output
"""

import argparse
import os
import sys
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

    R_pp_per_frame = [torch.eye(num_patches, device=device) for _ in range(T)]

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
    for layer_idx, attn_module in wrapped['spatial_attn']:
        attn_probs = attn_module.get_attn()       # [B*T, num_heads, N, N]
        attn_grad = attn_module.get_attn_gradients()  # [B*T, num_heads, N, N]
        
        if attn_probs is not None:
            for t in range(T):
                # --- identical to example.py lines 24-30, applied per frame ---
                cam = attn_probs[t]   # [num_heads, N, N]
                grad = attn_grad[t]   # [num_heads, N, N]
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                R_pp_per_frame[t] += torch.matmul(cam, R_pp_per_frame[t])
    
    # =========================================================================
    # Extract heatmaps via pooling head cross-attention (CLS-token analog).
    #
    # SigLIP has no CLS token. Its pooling head has a learnable probe that
    # cross-attends to all 196 patches — functionally identical to CLS.
    #
    # Original Chefer (ViT with CLS):   image_relevance = R[0, 1:]
    # Our equivalent (SigLIP, no CLS):  image_relevance = cam_cross @ R
    #
    # cam_cross is the gradient-weighted probe→patches attention from the
    # pooling head, so cam_cross @ R reads the probe's relevance row.
    # =========================================================================
    
    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()           # [B*T, num_heads, 1, N]
    head_grad = head_attn_module.get_attn_gradients()  # [B*T, num_heads, 1, N]
    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")
    
    heatmaps = []
    for t in range(T):
        # Gradient-weighted cross-attention (same rule as spatial layers)
        cam = head_attn[t]   # [num_heads, 1, N]
        grad = head_grad[t]  # [num_heads, 1, N]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)          # [1, N]
        
        # Since SigLIP has no CLS token, cam_cross @ R is the equivalent of R[cls, 1:] in original Chefer
        image_relevance = torch.matmul(cam, R_pp_per_frame[t]).squeeze(0)  # [N]
        
        heatmap = image_relevance.detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(heatmap)
    
    heatmaps = np.stack(heatmaps, axis=0)  # [T, patch_size, patch_size]

    # =========================================================================
    # Per-frame temporal weighting via R_tt accumulated through:
    #   (a) 12 backbone temporal_attn modules  -- per-spatial-position cam.
    #       attn shape: [B*N, H, T, T]. We keep the per-position structure by
    #       maintaining R_tt of shape [N, T, T] and using batched matmul.
    #   (b) 2 classifier head TransformerEncoder layers -- single global cam
    #       [T, T] applied (broadcast) to every spatial position.
    # Final per-frame weight: w[t] = R_tt.mean(dim=(0, 1))[t]  -- collapses
    # both the spatial-position axis and the output-row axis. Justified
    # because the head's mean-pool over T frames uniformly weights each
    # output position, and we average across spatial positions only at
    # extraction time (Option B of the temporal-R derivation).
    # =========================================================================
    temporal_weights = np.ones(T, dtype=np.float32)
    if wrapped['temporal_attn'] or wrapped['head_te_attn']:
        # Initialize per-spatial-position R_tt: [N, T, T]
        N_patches_check = num_patches
        R_tt = torch.eye(T, device=device, dtype=torch.float32)\
            .unsqueeze(0).expand(N_patches_check, T, T).contiguous()

        # (a) Backbone temporal layers -- per-spatial-position
        for layer_idx, t_attn in wrapped['temporal_attn']:
            attn = t_attn.get_attn()    # [B*N, H, T, T]
            grad = t_attn.get_attn_gradients()
            if attn is None or grad is None:
                print(f'  WARNING: backbone temporal layer {layer_idx} attn/grad not captured.')
                continue
            # B==1 inference: shape collapses to [N, H, T, T]
            attn = attn.float()
            grad = grad.float()
            cam = (grad * attn).clamp(min=0).mean(dim=1)   # [N, T, T] -- mean over heads
            R_tt = R_tt + torch.bmm(cam, R_tt)

        # (b) Head TE layers -- broadcast the same cam across all N positions
        for layer_idx, te_attn in wrapped['head_te_attn']:
            cam_h = te_attn.get_attn()    # [B, H, T, T]
            grad_h = te_attn.get_attn_gradients()
            if cam_h is None or grad_h is None:
                print(f'  WARNING: head TE layer {layer_idx} attn/grad not captured.')
                continue
            cam_h = cam_h.reshape(-1, T, T).float()
            grad_h = grad_h.reshape(-1, T, T).float()
            cam_global = (grad_h * cam_h).clamp(min=0).mean(dim=0)   # [T, T]
            R_tt = R_tt + cam_global @ R_tt   # broadcasts: [T,T] @ [N,T,T] -> [N,T,T]

        # Collapse to per-frame weights: average over spatial positions and output rows.
        w = R_tt.mean(dim=(0, 1)).detach().cpu().numpy()   # [T]
        if w.max() > 0:
            temporal_weights = (w / w.max()).astype(np.float32)
        print(f"  R_tt temporal weights (normalized, backbone+head): "
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
        description='Chefer explainability inference (per-frame spatial-only) for MatchVision classification.')
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_cluster.py',
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
    # [Step 3] Directory for saving per-clip saliency maps and scores.
    parser.add_argument('--saliency_save_dir', type=str, default=None,
                        help='Optional directory to save per-clip saliency maps and scores for convergence analysis')
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
                saliency_save_dir=getattr(args, 'saliency_save_dir', None),  # [Step 3]
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
