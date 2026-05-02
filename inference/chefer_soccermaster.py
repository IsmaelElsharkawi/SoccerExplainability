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

    Wraps all 27 spatial self_attn layers in VisionBackbone and the
    pooling head cross-attention. Temporal attention is NOT wrapped.
    """
    wrapped = {
        'spatial_attn': [],
        'pooling_head_attn': None,
    }

    vision_backbone = model.backbone.vision_model

    for i, block in enumerate(vision_backbone.encoder_blocks):
        wrapped_spatial = wrap_siglip_attention_module(block.encoder.self_attn)
        wrapped['spatial_attn'].append((i, wrapped_spatial))

    wrapped['pooling_head_attn'] = wrap_pooling_head_attention(vision_backbone.head.attention)

    print(f"Wrapped attention modules (SoccerMaster per-frame spatial-only):")
    print(f"  - Spatial (SigLIP2): {len(wrapped['spatial_attn'])} layers")
    print(f"  - Pooling head (probe->patches): yes")

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

    R_pp_per_frame = [torch.eye(num_patches, device=device, dtype=torch.float16) for _ in range(T)]

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
    # Relevance propagation
    # =========================================================================
    for layer_idx, attn_module in wrapped['spatial_attn']:
        attn_probs = attn_module.get_attn()       # [B*T, num_heads, N, N]
        attn_grad = attn_module.get_attn_gradients()  # [B*T, num_heads, N, N]

        if attn_probs is not None:
            for t in range(T):
                cam = attn_probs[t]   # [num_heads, N, N]
                grad = attn_grad[t]   # [num_heads, N, N]
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                R_pp_per_frame[t] += torch.matmul(cam, R_pp_per_frame[t])

    # =========================================================================
    # Extract heatmaps via pooling head cross-attention
    # =========================================================================
    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()           # [B*T, num_heads, 1, N]
    head_grad = head_attn_module.get_attn_gradients()  # [B*T, num_heads, 1, N]
    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")

    heatmaps = []
    for t in range(T):
        cam = head_attn[t]   # [num_heads, 1, N]
        grad = head_grad[t]  # [num_heads, 1, N]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)          # [1, N]

        image_relevance = torch.matmul(cam, R_pp_per_frame[t]).squeeze(0)  # [N]

        heatmap = image_relevance.detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(heatmap)

    heatmaps = np.stack(heatmaps, axis=0).astype(np.float32)  # [T, patch_size, patch_size]

    # Min-max normalization per frame
    for t in range(T):
        h = heatmaps[t]
        if h.max() > h.min():
            heatmaps[t] = (h - h.min()) / (h.max() - h.min())
        else:
            heatmaps[t] = np.zeros_like(h)

    return heatmaps


# ============================================================================
# Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chefer explainability inference (per-frame spatial-only) for SoccerMaster classification.')
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_cluster.py',
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
