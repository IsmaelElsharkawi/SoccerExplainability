import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from model.MatchVision_classifier import MatchVision_Classifier

from coco_attribution_eval import CocoAttributionEvaluator


# ============================================================================
# Config helpers
# ============================================================================

def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config


def reshape_transform(result, height=14, width=14, timesteps=30):
    # Result shape: [batch* time, 196, embedding_dim]
    BT, N, C = result.shape
    print("Tensor Shape:", result.shape)
    result = result.unsqueeze(0)
    result = result.reshape(BT // timesteps, timesteps, height, width, C)

    # Transpose dimensions to get [batch, embedding_dim, time, height, width].
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
                },
                _f,
                indent=2,
                default=_conv,
            )
        print(f'  Detailed results saved to {eval_output_path}')
