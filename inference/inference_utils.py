import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from model.MatchVision.MatchVision_classifier import MatchVision_Classifier

from coco_attribution_eval import CocoAttributionEvaluator


# ============================================================================
# Chefer Method Constants
# ============================================================================

LABEL_NAMES = [
    'var', 'end of half game', 'clearance', 'second yellow card',
    'injury', 'ball possession', 'throw in', 'show added time',
    'shot off target', 'start of half game', 'substitution',
    'saved by goal-keeper', 'red card', 'lead to corner',
    'ball out of play', 'off side', 'goal', 'penalty',
    'yellow card', 'foul lead to penalty', 'corner', 'free kick',
    'foul with no card'
]

LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}


# ============================================================================
# Chefer Attention Wrapping
# ============================================================================

def wrap_siglip_attention_module(attn_module):
    """
    Wrap a SigLIP Attention module to capture attention weights AND gradients.

    Replaces the forward method to manually compute multi-head attention so we
    can store the attention matrix and register a gradient hook on it.

    Works for both SigLIP-base (MatchVision) and SigLIP2-large (SoccerMaster)
    since both use the same SiglipAttention interface (q_proj, k_proj, v_proj, out_proj).

    Args:
        attn_module: SiglipAttention module

    Returns:
        The same module, now with get_attn() and get_attn_gradients() methods
    """
    if hasattr(attn_module, '_chefer_wrapped'):
        return attn_module

    embed_dim = attn_module.q_proj.out_features
    num_heads = attn_module.num_heads
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    attn_storage = {}
    grad_storage = {}

    def wrapped_forward(hidden_states, attention_mask=None, output_attentions=False):
        B, N, C = hidden_states.shape

        query = attn_module.q_proj(hidden_states)
        key = attn_module.k_proj(hidden_states)
        value = attn_module.v_proj(hidden_states)

        query = query.view(B, N, num_heads, head_dim).transpose(1, 2)
        key = key.view(B, N, num_heads, head_dim).transpose(1, 2)
        value = value.view(B, N, num_heads, head_dim).transpose(1, 2)

        attn = (query @ key.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        attn_storage['attn'] = attn.detach()

        if attn.requires_grad:
            def grad_hook(grad):
                grad_storage['grad'] = grad.detach()
            attn.register_hook(grad_hook)

        out = (attn @ value).transpose(1, 2).reshape(B, N, C)
        out = attn_module.out_proj(out)

        if output_attentions:
            return out, attn
        return out, None

    attn_module.forward = wrapped_forward

    attn_module.get_attn = lambda: attn_storage.get('attn')
    attn_module.get_attn_gradients = lambda: grad_storage.get('grad')
    attn_module._chefer_wrapped = True

    return attn_module


def wrap_pooling_head_attention(mha_module):
    """
    Wrap SigLIP's pooling head nn.MultiheadAttention for cross-attention capture.

    The pooling head does: output = attention(probe, patches, patches)
    where probe is a learnable CLS-like token and patches are the spatial features.

    Works for both SigLIP-base (MatchVision) and SigLIP2-large (SoccerMaster).

    Args:
        mha_module: nn.MultiheadAttention (batch_first=True)

    Returns:
        The same module, now with get_attn() and get_attn_gradients() methods
    """
    if hasattr(mha_module, '_chefer_wrapped'):
        return mha_module

    embed_dim = mha_module.embed_dim
    num_heads = mha_module.num_heads
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    attn_storage = {}
    grad_storage = {}

    def wrapped_forward(query, key, value, **kwargs):
        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        w = mha_module.in_proj_weight
        b = mha_module.in_proj_bias

        Q = F.linear(query, w[:embed_dim], b[:embed_dim])
        K = F.linear(key, w[embed_dim:2*embed_dim], b[embed_dim:2*embed_dim])
        V = F.linear(value, w[2*embed_dim:], b[2*embed_dim:])

        Q = Q.view(B, Nq, num_heads, head_dim).transpose(1, 2)
        K = K.view(B, Nk, num_heads, head_dim).transpose(1, 2)
        V = V.view(B, Nk, num_heads, head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        attn_storage['attn'] = attn.detach()
        if attn.requires_grad:
            def grad_hook(grad):
                grad_storage['grad'] = grad.detach()
            attn.register_hook(grad_hook)

        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)
        out = mha_module.out_proj(out)

        return out, attn

    mha_module.forward = wrapped_forward

    mha_module.get_attn = lambda: attn_storage.get('attn')
    mha_module.get_attn_gradients = lambda: grad_storage.get('grad')
    mha_module._chefer_wrapped = True

    return mha_module


# ============================================================================
# Chefer Attribution Renderer
# ============================================================================

def chefer_attribution_renderer(frame_float, attribution_map):
    """
    Render a Chefer heatmap overlay on a frame.

    Args:
        frame_float: [H, W, 3] float32 image in [0, 1]
        attribution_map: [h, w] float32 heatmap in [0, 1] (may differ in size)

    Returns:
        [H, W, 3] uint8 RGB overlay image
    """
    H, W = frame_float.shape[:2]
    heatmap = cv2.resize(attribution_map, (W, H), interpolation=cv2.INTER_LINEAR)

    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    # Magenta colormap: black (low) → magenta (high).  R=h, G=0, B=h in RGB.
    zeros = np.zeros_like(heatmap_uint8)
    heatmap_colored = cv2.merge([heatmap_uint8, zeros, heatmap_uint8]).astype(np.float32) / 255.0  # RGB

    alpha = 0.5
    overlay = alpha * heatmap_colored + (1 - alpha) * frame_float
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


# ============================================================================
# Config helpers
# ============================================================================

def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config

# For SoccerMaster
# def reshape_transform(result, height=14, width=14, timesteps=30):
#     # Result shape is typically [batch*time, 196, embedding_dim] for video
#     # and [batch, 196, embedding_dim] for image-only SigLIP.
#     BT, C, N = result.shape
#     print("Tensor Shape:", result.shape)

#     if N != height * width:
#         raise ValueError(
#             f"Unexpected token count {N}; expected {height * width} for {height}x{width} patches."
#         )

#     effective_timesteps = timesteps if BT >= timesteps and BT % timesteps == 0 else 1
#     batch_size = BT // effective_timesteps
#     result = result.reshape(BT, C, height, width).unsqueeze(0)
#     print(result.shape)
#     #result = result.reshape(batch_size, C, effective_timesteps, height, width, C)

#     # Transpose dimensions to [batch, embedding_dim, time, height, width].
#     result = result.permute(0, 2, 1, 3, 4)
#     print("Reshaped Tensor Shape:", result.shape)
#     return result

def reshape_transform(result, height=14, width=14, timesteps=30):
    # Result shape is typically [batch*time, 196, embedding_dim] for video
    # and [batch, 196, embedding_dim] for image-only SigLIP.
    BT, N, C = result.shape
    print("Tensor Shape:", result.shape)

    if N != height * width:
        raise ValueError(
            f"Unexpected token count {N}; expected {height * width} for {height}x{width} patches."
        )

    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}.")

    if BT % timesteps != 0:
        raise ValueError(
            f"Cannot reshape activations with leading dimension {BT} into batches of {timesteps} timesteps."
        )

    batch_size = BT // timesteps

    result = result.reshape(batch_size, timesteps, height, width, C)

    # Transpose dimensions to [batch, embedding_dim, time, height, width].
    result = result.permute(0, 4, 1, 2, 3)
    print("Reshaped Tensor Shape:", result.shape)
    return result


# ============================================================================
# Dataset
# ============================================================================

def create_test_dataloader(config_test_dataset):
    """Create test dataset and DataLoader from the test config dict."""
    if config_test_dataset['balanced_or_not'] == 'balanced':
        test_dataset_type = VideoCaptionDataset_Balanced
    else:
        test_dataset_type = VideoCaptionDataset

    test_dataset = test_dataset_type(
        json_file=config_test_dataset['json'],
        video_base_dir=config_test_dataset['video_base'],
        sample=config_test_dataset['sample'],
        keywords=config_test_dataset['keywords'],
        processor_model_name=config_test_dataset.get('processor_model_name', 'google/siglip-base-patch16-224'),
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_test_dataset['batch_size'],
        num_workers=config_test_dataset['num_workers'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return test_dataset, test_data_loader


# ============================================================================
# Model
# ============================================================================

def load_classifier(config_test_dataset, classifier_transformer_type, encoder_type,
                    use_transformer, checkpoint_path, devices, device_ids):
    """Create a MatchVision_Classifier, load checkpoint, and wrap in DataParallel."""
    classifier = MatchVision_Classifier(
        keywords=config_test_dataset['keywords'],
        classifier_transformer_type=classifier_transformer_type,
        vision_encoder_type=encoder_type,
        use_transformer=use_transformer,
    ).eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {key.replace('module.', ''): value
                      for key, value in checkpoint['state_dict'].items()}
    classifier.load_state_dict(new_state_dict)

    classifier = classifier.to(devices[0])
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
    return classifier


# ============================================================================
# COCO Attribution Evaluation
# ============================================================================

def setup_attribution_evaluator(coco_json_path):
    """Create a CocoAttributionEvaluator if the annotation file exists."""
    attribution_evaluator = None
    all_eval_results = {}
    if coco_json_path and os.path.exists(coco_json_path):
        attribution_evaluator = CocoAttributionEvaluator(coco_json_path)
        print(f"Loaded COCO annotations from {coco_json_path} "
              f"({len(attribution_evaluator.coco['annotations'])} annotations, "
              f"{len(attribution_evaluator.coco['images'])} images)")
    elif coco_json_path:
        print(f'Warning: COCO JSON not found at {coco_json_path}, skipping evaluation.')
    return attribution_evaluator, all_eval_results


def match_video_id(attribution_evaluator, video_path):
    """Find the COCO annotation video ID that matches the given video path."""
    if attribution_evaluator is None:
        return None
    for ann_vid in attribution_evaluator.get_annotated_video_ids():
        if video_path.endswith(ann_vid) or ann_vid in video_path:
            return ann_vid
    return None


def evaluate_and_print_video(attribution_evaluator, heatmaps, matched_video_id,
                             video_name, cam_threshold, all_eval_results):
    """Run per-video COCO attribution evaluation and print results."""
    if attribution_evaluator is None or matched_video_id is None:
        return
    eval_result = attribution_evaluator.evaluate_video(
        heatmaps,
        matched_video_id,
        start_second=0,
        cam_threshold=cam_threshold,
    )
    all_eval_results[matched_video_id] = eval_result
    es = eval_result['summary']
    print(f"  [COCO Eval] {video_name}: "
          f"Energy={es['mean_energy_inside_bbox']:.3f}  "
          f"Pointing={es['mean_pointing_accuracy']:.3f}  "
          f"IoU={es['mean_iou']:.3f}  "
          f"({es['annotated_frames']}/{es['total_frames']} annotated frames)")
    group_scores = es.get('label_group_scores', {})
    for group_name in ['small_only', 'small_large', 'small_large_visual_cues']:
        gs = group_scores.get(group_name)
        if not gs:
            continue
        print(
            f"    - {group_name}: "
            f"Energy={gs['mean_energy_inside_bbox']:.3f}  "
            f"Pointing={gs['mean_pointing_accuracy']:.3f}  "
            f"IoU={gs['mean_iou']:.3f}  "
            f"({gs['annotated_frames']} frames)"
        )

    temporal_scores = es.get('temporal_localization', {})
    tier_scores = temporal_scores.get('tiers', {})
    if temporal_scores:
        print(
            f"    - temporal mean: "
            f"mean_tIoU={temporal_scores.get('mean_tIoU', float('nan')):.3f}  "
            f"(thr_ratio={temporal_scores.get('score_threshold_ratio', float('nan')):.2f})"
        )
        for tier_name in ['small_only', 'small_large', 'small_large_visual_cues']:
            ts = tier_scores.get(tier_name)
            if not ts:
                continue
            print(
                f"      * {tier_name}: "
                f"tIoU={ts['tIoU']:.3f}  "
                f"(gt={ts['gt_frames']}, pred={ts['pred_frames']})"
            )


def print_and_save_eval_summary(all_eval_results, eval_output_path=None,
                                summary_title='Attribution Label-Group Evaluation Summary'):
    """Print global evaluation summary and optionally save to JSON."""
    if not all_eval_results:
        return

    print(f'\n===== {summary_title} =====')
    group_video_means = {
        'small_only': {'energy': [], 'pointing': [], 'iou': [], 'frames': 0},
        'small_large': {'energy': [], 'pointing': [], 'iou': [], 'frames': 0},
        'small_large_visual_cues': {'energy': [], 'pointing': [], 'iou': [], 'frames': 0},
    }
    temporal_video_means = {
        'small_only': {'tiou': [], 'gt_frames': 0},
        'small_large': {'tiou': [], 'gt_frames': 0},
        'small_large_visual_cues': {'tiou': [], 'gt_frames': 0},
    }
    videos_with_any_group = 0
    for _, res in all_eval_results.items():
        s = res['summary']
        ggs = s.get('label_group_scores', {})
        has_group_data = False
        for group_name in group_video_means:
            gs = ggs.get(group_name)
            if not gs:
                continue
            if not np.isnan(gs['mean_energy_inside_bbox']):
                group_video_means[group_name]['energy'].append(gs['mean_energy_inside_bbox'])
                has_group_data = True
            if not np.isnan(gs['mean_pointing_accuracy']):
                group_video_means[group_name]['pointing'].append(gs['mean_pointing_accuracy'])
            if not np.isnan(gs['mean_iou']):
                group_video_means[group_name]['iou'].append(gs['mean_iou'])
            group_video_means[group_name]['frames'] += int(gs.get('annotated_frames', 0))
        if has_group_data:
            videos_with_any_group += 1

        temporal_scores = s.get('temporal_localization', {})
        temporal_tiers = temporal_scores.get('tiers', {})
        for tier_name in temporal_video_means:
            ts = temporal_tiers.get(tier_name)
            if not ts:
                continue
            if not np.isnan(ts.get('tIoU', np.nan)):
                temporal_video_means[tier_name]['tiou'].append(float(ts['tIoU']))
            temporal_video_means[tier_name]['gt_frames'] += int(ts.get('gt_frames', 0))

    print(f'  Videos evaluated: {videos_with_any_group}')
    global_group_summary = {}
    for group_name, vals in group_video_means.items():
        if vals['energy']:
            global_group_summary[group_name] = {
                'mean_energy_inside_bbox': float(np.mean(vals['energy'])),
                'mean_pointing_accuracy': float(np.mean(vals['pointing'])),
                'mean_iou': float(np.mean(vals['iou'])),
                'annotated_frames': int(vals['frames']),
            }
            print(
                f"  {group_name}: "
                f"Energy={global_group_summary[group_name]['mean_energy_inside_bbox']:.4f}  "
                f"Pointing={global_group_summary[group_name]['mean_pointing_accuracy']:.4f}  "
                f"IoU={global_group_summary[group_name]['mean_iou']:.4f}  "
                f"(frames={global_group_summary[group_name]['annotated_frames']})"
            )
        else:
            global_group_summary[group_name] = {
                'mean_energy_inside_bbox': float('nan'),
                'mean_pointing_accuracy': float('nan'),
                'mean_iou': float('nan'),
                'annotated_frames': 0,
            }
            print(f'  {group_name}: no matching annotated frames')

    global_temporal_summary = {}
    print('  Temporal localization (tiers):')
    for tier_name, vals in temporal_video_means.items():
        if vals['tiou']:
            global_temporal_summary[tier_name] = {
                'mean_tIoU': float(np.mean(vals['tiou'])),
                'gt_frames': int(vals['gt_frames']),
            }
            print(
                f"    {tier_name}: "
                f"mean_tIoU={global_temporal_summary[tier_name]['mean_tIoU']:.4f}  "
                f"(gt_frames={global_temporal_summary[tier_name]['gt_frames']})"
            )
        else:
            global_temporal_summary[tier_name] = {
                'mean_tIoU': float('nan'),
                'gt_frames': 0,
            }
            print(f'    {tier_name}: no temporal GT-positive frames')

    valid_global_tious = [
        vals['mean_tIoU'] for vals in global_temporal_summary.values()
        if not np.isnan(vals['mean_tIoU'])
    ]
    global_temporal_means = {
        'mean_tIoU_across_tiers': float(np.mean(valid_global_tious)) if valid_global_tious else float('nan'),
    }
    print(
        f"  Temporal means across tiers: "
        f"mean_tIoU={global_temporal_means['mean_tIoU_across_tiers']:.4f}"
    )

    if eval_output_path:
        def _conv(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(eval_output_path, 'w') as _f:
            json.dump(
                {
                    'per_video': {k: v for k, v in all_eval_results.items()},
                    'global_label_group_summary': global_group_summary,
                    'global_temporal_localization_summary': {
                        'tiers': global_temporal_summary,
                        'means': global_temporal_means,
                    },
                },
                _f,
                indent=2,
                default=_conv,
            )
        print(f'  Detailed results saved to {eval_output_path}')
