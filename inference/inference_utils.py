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


def wrap_temporal_attention_module(mha_module):
    """
    Wrap a backbone temporal nn.MultiheadAttention(batch_first=True) for Chefer.

    The underlying module is structurally identical to the SigLIP pooling head
    (both are nn.MultiheadAttention), so we delegate to wrap_pooling_head_attention.
    Renamed for call-site clarity in chefer_matchvision/soccermaster scripts.

    Captures attention of shape [B*N, H, T, T] -- one [T, T] attention per
    spatial position, since MatchVision's temporal_attn is called on tensors
    rearranged via 'b t n m -> (b n) t m' (each spatial position attended over
    time independently).
    """
    return wrap_pooling_head_attention(mha_module)


def wrap_transformer_encoder_layer(te_layer):
    """
    Wrap an nn.TransformerEncoderLayer's self-attention for Chefer relevance.

    The underlying nn.MultiheadAttention is structurally identical to the
    SigLIP pooling head, so we use the same wrapping pattern. Differences:
      - Self-attention (q == k == v == sequence), not cross-attention.
      - Convention can be batch_first=True (SoccerMaster) or False (MatchVision);
        we read `self_attn.batch_first` and transpose accordingly.

    The wrapped module exposes `get_attn()` / `get_attn_gradients()` returning
    tensors of shape [B, num_heads, T, T] regardless of batch_first.

    Returns the inner self-attention module (same convention as the other
    wrappers in this file).
    """
    self_attn = te_layer.self_attn

    if hasattr(self_attn, '_chefer_wrapped'):
        return self_attn

    embed_dim = self_attn.embed_dim
    num_heads = self_attn.num_heads
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    batch_first = getattr(self_attn, 'batch_first', False)

    attn_storage = {}
    grad_storage = {}

    def wrapped_forward(query, key, value, **kwargs):
        if not batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        w = self_attn.in_proj_weight
        b = self_attn.in_proj_bias

        Q = F.linear(query, w[:embed_dim], None if b is None else b[:embed_dim])
        K = F.linear(key, w[embed_dim:2*embed_dim],
                     None if b is None else b[embed_dim:2*embed_dim])
        V = F.linear(value, w[2*embed_dim:],
                     None if b is None else b[2*embed_dim:])

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
        out = self_attn.out_proj(out)

        if not batch_first:
            out = out.transpose(0, 1)

        return out, attn

    self_attn.forward = wrapped_forward
    self_attn.get_attn = lambda: attn_storage.get('attn')
    self_attn.get_attn_gradients = lambda: grad_storage.get('grad')
    self_attn._chefer_wrapped = True

    return self_attn


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
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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


def match_video_ids(attribution_evaluator, video_path):
    """Find all COCO annotation video IDs that match the given video path.

    A single video file may appear as multiple annotation variants (e.g.
    ``2_45_08.mp4`` and ``2_45_08.mp4#penalty``).  Returns a list of all
    matching ``video_id`` strings, with the exact (non-fragment) match first.

    If *video_path* contains a ``#fragment`` (e.g. from the dataset's
    ``variant`` field), only the COCO video ID that ends with that exact
    fragment is returned.
    """
    if attribution_evaluator is None:
        return []

    # Split off an optional #variant fragment appended by the dataset.
    if '#' in video_path:
        base_path, fragment = video_path.rsplit('#', 1)
    else:
        base_path, fragment = video_path, None

    matches = []
    for ann_vid in attribution_evaluator.get_annotated_video_ids():
        if base_path.endswith(ann_vid) or ann_vid in base_path:
            # Base path (without fragment) is a substring match.
            if fragment is None:
                # No variant requested — match only the base annotation
                # (the one whose video_id does NOT contain a '#').
                if '#' not in ann_vid:
                    matches.append(ann_vid)
            else:
                # Variant requested — match only the annotation whose
                # video_id ends with the same fragment.
                if ann_vid.endswith(f'#{fragment}'):
                    matches.append(ann_vid)
        elif fragment is not None and (base_path.endswith(ann_vid.split('#')[0]) or ann_vid.split('#')[0] in base_path):
            # The COCO video_id itself contains a fragment — check if
            # the base file path matches and the fragment matches.
            if ann_vid.endswith(f'#{fragment}'):
                matches.append(ann_vid)

    matches.sort(key=lambda v: ('#' in v, v))
    return matches


# Keep a thin wrapper so old call-sites that only need a single id still work.
def match_video_id(attribution_evaluator, video_path):
    """Return the first matching video ID, or *None*."""
    ids = match_video_ids(attribution_evaluator, video_path)
    return ids[0] if ids else None


def _sanitize_clip_id(clip_id):
    """[Step 3] Make a filesystem-safe filename stem from a clip id."""
    return ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in str(clip_id))


def save_clip_saliency_and_scores(save_dir, clip_id, video_name, heatmaps,
                                  eval_result, attribution_scores=None,
                                  prediction_text=None, ground_truth_text=None):
    """[Step 3] Persist per-clip saliency maps and scores for later analysis.

    Writes two files per clip into ``save_dir``:
      * ``<clip>.npz``  -- saliency maps + per-frame score arrays (compressed).
      * ``<clip>.json`` -- the full eval summary plus lightweight metadata.

    These artifacts are consumed by ``convergence_analysis.py`` (Step 4) so the
    convergence study and per-class tables can be built without re-running the
    attribution models.
    """
    os.makedirs(save_dir, exist_ok=True)
    stem = _sanitize_clip_id(clip_id)

    heatmaps_np = np.asarray(heatmaps, dtype=np.float32)
    num_frames = heatmaps_np.shape[0]

    # Per-frame mean intensity: the temporal saliency signal used by T-* metrics.
    frame_scores = np.mean(heatmaps_np.reshape(num_frames, -1), axis=1)

    # Per-frame spatial metric arrays (NaN where a frame has no ROI/annotation).
    energy_arr, pointing_arr, iou_arr, sauc_arr = [], [], [], []
    for frame_item in eval_result.get('per_frame', []):
        fm = frame_item.get('metrics') or {}
        energy_arr.append(fm.get('energy_inside_bbox', np.nan))
        pointing_arr.append(fm.get('pointing_accuracy', np.nan))
        iou_arr.append(fm.get('iou', np.nan))
        sauc_arr.append(fm.get('s_auc', np.nan))

    npz_payload = {
        'saliency_maps': heatmaps_np,
        'frame_scores': frame_scores.astype(np.float32),
        'frame_energy': np.asarray(energy_arr, dtype=np.float32),
        'frame_pointing': np.asarray(pointing_arr, dtype=np.float32),
        'frame_s_iou': np.asarray(iou_arr, dtype=np.float32),
        'frame_s_auc': np.asarray(sauc_arr, dtype=np.float32),
    }
    if attribution_scores is not None:
        npz_payload['attribution_scores'] = np.asarray(attribution_scores, dtype=np.float32)

    np.savez_compressed(os.path.join(save_dir, f'{stem}.npz'), **npz_payload)

    def _conv(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    meta = {
        'clip_id': str(clip_id),
        'video_name': str(video_name),
        'num_frames': int(num_frames),
        'prediction_text': prediction_text,
        'ground_truth_text': ground_truth_text,
        'summary': eval_result.get('summary', {}),
    }
    with open(os.path.join(save_dir, f'{stem}.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=_conv)


def evaluate_and_print_video(attribution_evaluator, heatmaps, matched_video_ids,
                             video_name, cam_threshold, all_eval_results,
                             saliency_save_dir=None, attribution_scores=None,
                             prediction_text=None, ground_truth_text=None):
    """Run per-video COCO attribution evaluation and print results.

    ``matched_video_ids`` may be a single string, a list of strings, or
    *None*.  Each variant is evaluated independently.

    [Step 3] When ``saliency_save_dir`` is provided, the raw saliency maps and
    per-clip scores are persisted to disk (one artifact bundle per clip) so
    that a later convergence analysis can reload them without re-running the
    models.
    """
    if attribution_evaluator is None or matched_video_ids is None:
        return

    # Normalise to a list so callers can pass a single id or a list.
    if isinstance(matched_video_ids, str):
        matched_video_ids = [matched_video_ids]

    for matched_video_id in matched_video_ids:
        eval_result = attribution_evaluator.evaluate_video(
            heatmaps,
            matched_video_id,
            start_second=0,
            cam_threshold=cam_threshold,
        )
        all_eval_results[matched_video_id] = eval_result

        # [Step 3] Persist saliency maps + scores for this clip.
        if saliency_save_dir is not None:
            save_clip_saliency_and_scores(
                saliency_save_dir,
                clip_id=matched_video_id,
                video_name=video_name,
                heatmaps=heatmaps,
                eval_result=eval_result,
                attribution_scores=attribution_scores,
                prediction_text=prediction_text,
                ground_truth_text=ground_truth_text,
            )

        es = eval_result['summary']
        print(f"  [COCO Eval] {video_name} ({matched_video_id}): "
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
                f"S-AUC={gs.get('mean_s_auc', float('nan')):.3f}  "
                f"({gs['annotated_frames']} frames)"
            )

        temporal_scores = es.get('temporal_localization', {})
        tier_scores = temporal_scores.get('tiers', {})
        if temporal_scores:
            print(
                f"    - temporal mean: "
                f"mean_tIoU={temporal_scores.get('mean_tIoU', float('nan')):.3f}  "
                f"mean_tIoU_sweep={temporal_scores.get('mean_tIoU_sweep', float('nan')):.3f}  "
                f"mean_tAUC={temporal_scores.get('mean_tAUC', float('nan')):.3f}  "
                f"mean_tAP={temporal_scores.get('mean_tAP', float('nan')):.3f}  "
                f"(thr_ratio={temporal_scores.get('score_threshold_ratio', float('nan')):.2f})"
            )
            for tier_name in ['small_only', 'small_large', 'small_large_visual_cues']:
                ts = tier_scores.get(tier_name)
                if not ts:
                    continue
                print(
                    f"      * {tier_name}: "
                    f"tIoU={ts['tIoU']:.3f}  "
                    f"tAUC={ts.get('tAUC', float('nan')):.3f}  "
                    f"tAP={ts.get('tAP', float('nan')):.3f}  "
                    f"(gt={ts['gt_frames']}, pred={ts['pred_frames']})"
                )


def print_and_save_eval_summary(all_eval_results, eval_output_path=None,
                                summary_title='Attribution Label-Group Evaluation Summary'):
    """Print global evaluation summary and optionally save to JSON."""
    if not all_eval_results:
        return

    print(f'\n===== {summary_title} =====')
    group_video_means = {
        'small_only': {'energy': [], 'pointing': [], 'iou': [], 's_auc': [], 'frames': 0},
        'small_large': {'energy': [], 'pointing': [], 'iou': [], 's_auc': [], 'frames': 0},
        'small_large_visual_cues': {'energy': [], 'pointing': [], 'iou': [], 's_auc': [], 'frames': 0},
    }
    temporal_video_means = {
        'small_only': {'tiou': [], 'tiou_sweep': [], 'tauc': [], 'tap': [], 'sweep_per_threshold': {}, 'gt_frames': 0},
        'small_large': {'tiou': [], 'tiou_sweep': [], 'tauc': [], 'tap': [], 'sweep_per_threshold': {}, 'gt_frames': 0},
        'small_large_visual_cues': {'tiou': [], 'tiou_sweep': [], 'tauc': [], 'tap': [], 'sweep_per_threshold': {}, 'gt_frames': 0},
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
            if not np.isnan(gs.get('mean_s_auc', np.nan)):  # [Step 2]
                group_video_means[group_name]['s_auc'].append(gs['mean_s_auc'])
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
            sweep = ts.get('tIoU_sweep')
            if sweep and not np.isnan(sweep.get('mean_over_thresholds', np.nan)):
                temporal_video_means[tier_name]['tiou_sweep'].append(float(sweep['mean_over_thresholds']))
                # Keep each threshold's IoU so it can be reported (averaged over clips).
                for ratio, tiou_val in (sweep.get('per_threshold', {}) or {}).items():
                    if not np.isnan(tiou_val):
                        temporal_video_means[tier_name]['sweep_per_threshold'].setdefault(ratio, []).append(float(tiou_val))
            if not np.isnan(ts.get('tAUC', np.nan)):
                temporal_video_means[tier_name]['tauc'].append(float(ts['tAUC']))
            if not np.isnan(ts.get('tAP', np.nan)):
                temporal_video_means[tier_name]['tap'].append(float(ts['tAP']))
            temporal_video_means[tier_name]['gt_frames'] += int(ts.get('gt_frames', 0))

    print(f'  Videos evaluated: {videos_with_any_group}')
    global_group_summary = {}
    for group_name, vals in group_video_means.items():
        if vals['energy']:
            global_group_summary[group_name] = {
                'mean_energy_inside_bbox': float(np.mean(vals['energy'])),
                'mean_pointing_accuracy': float(np.mean(vals['pointing'])),
                'mean_iou': float(np.mean(vals['iou'])),
                'mean_s_auc': float(np.mean(vals['s_auc'])) if vals['s_auc'] else float('nan'),  # [Step 2]
                'annotated_frames': int(vals['frames']),
            }
            print(
                f"  {group_name}: "
                f"Energy={global_group_summary[group_name]['mean_energy_inside_bbox']:.4f}  "
                f"Pointing={global_group_summary[group_name]['mean_pointing_accuracy']:.4f}  "
                f"IoU={global_group_summary[group_name]['mean_iou']:.4f}  "
                f"S-AUC={global_group_summary[group_name]['mean_s_auc']:.4f}  "
                f"(frames={global_group_summary[group_name]['annotated_frames']})"
            )
        else:
            global_group_summary[group_name] = {
                'mean_energy_inside_bbox': float('nan'),
                'mean_pointing_accuracy': float('nan'),
                'mean_iou': float('nan'),
                'mean_s_auc': float('nan'),  # [Step 2]
                'annotated_frames': 0,
            }
            print(f'  {group_name}: no matching annotated frames')

    global_temporal_summary = {}
    print('  Temporal localization (tiers):')
    for tier_name, vals in temporal_video_means.items():
        if vals['tiou']:
            global_temporal_summary[tier_name] = {
                'mean_tIoU': float(np.mean(vals['tiou'])),
                'mean_tIoU_sweep': float(np.mean(vals['tiou_sweep'])) if vals['tiou_sweep'] else float('nan'),
                'mean_tIoU_per_threshold': {
                    ratio: float(np.mean(tious))
                    for ratio, tious in sorted(vals['sweep_per_threshold'].items())
                },
                'mean_tAUC': float(np.mean(vals['tauc'])) if vals['tauc'] else float('nan'),
                'mean_tAP': float(np.mean(vals['tap'])) if vals['tap'] else float('nan'),
                'gt_frames': int(vals['gt_frames']),
            }
            print(
                f"    {tier_name}: "
                f"mean_tIoU={global_temporal_summary[tier_name]['mean_tIoU']:.4f}  "
                f"mean_tIoU_sweep={global_temporal_summary[tier_name]['mean_tIoU_sweep']:.4f}  "
                f"mean_tAUC={global_temporal_summary[tier_name]['mean_tAUC']:.4f}  "
                f"mean_tAP={global_temporal_summary[tier_name]['mean_tAP']:.4f}  "
                f"(gt_frames={global_temporal_summary[tier_name]['gt_frames']})"
            )
            if global_temporal_summary[tier_name]['mean_tIoU_per_threshold']:
                per_thr = '  '.join(
                    f"{ratio}={val:.4f}"
                    for ratio, val in global_temporal_summary[tier_name]['mean_tIoU_per_threshold'].items()
                )
                print(f"        per-threshold tIoU: {per_thr}")
        else:
            global_temporal_summary[tier_name] = {
                'mean_tIoU': float('nan'),
                'mean_tIoU_sweep': float('nan'),
                'mean_tIoU_per_threshold': {},
                'mean_tAUC': float('nan'),
                'mean_tAP': float('nan'),
                'gt_frames': 0,
            }
            print(f'    {tier_name}: no temporal GT-positive frames')

    valid_global_tious = [
        vals['mean_tIoU'] for vals in global_temporal_summary.values()
        if not np.isnan(vals['mean_tIoU'])
    ]
    valid_global_sweeps = [
        vals['mean_tIoU_sweep'] for vals in global_temporal_summary.values()
        if not np.isnan(vals['mean_tIoU_sweep'])
    ]
    valid_global_taucs = [
        vals['mean_tAUC'] for vals in global_temporal_summary.values()
        if not np.isnan(vals['mean_tAUC'])
    ]
    valid_global_taps = [
        vals['mean_tAP'] for vals in global_temporal_summary.values()
        if not np.isnan(vals['mean_tAP'])
    ]
    global_temporal_means = {
        'mean_tIoU_across_tiers': float(np.mean(valid_global_tious)) if valid_global_tious else float('nan'),
        'mean_tIoU_sweep_across_tiers': float(np.mean(valid_global_sweeps)) if valid_global_sweeps else float('nan'),
        'mean_tAUC_across_tiers': float(np.mean(valid_global_taucs)) if valid_global_taucs else float('nan'),
        'mean_tAP_across_tiers': float(np.mean(valid_global_taps)) if valid_global_taps else float('nan'),
    }
    print(
        f"  Temporal means across tiers: "
        f"mean_tIoU={global_temporal_means['mean_tIoU_across_tiers']:.4f}  "
        f"mean_tIoU_sweep={global_temporal_means['mean_tIoU_sweep_across_tiers']:.4f}  "
        f"mean_tAUC={global_temporal_means['mean_tAUC_across_tiers']:.4f}  "
        f"mean_tAP={global_temporal_means['mean_tAP_across_tiers']:.4f}"
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
