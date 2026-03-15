import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/content/SoccerExplainability')

from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from model.MatchVision_classifier import MatchVision_Classifier
from model.SigLIP_classifier import SigLIP_Classifier
from config.model_type import MODEL_TYPE
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from coco_attribution_eval import CocoAttributionEvaluator
from inference_utils import load_config, reshape_transform
from visualization_video import save_lowres_visualization_video


base_path = '/content/drive/MyDrive/arsenal-paris-gradcam/'
high_res_video_path = '/content/drive/MyDrive/arsenal-paris-high-res/2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG/'


def get_gradcam_target_layer(model):
    """Resolve a SigLIP-compatible target layer for Grad-CAM.

    Supports both custom wrappers and raw Hugging Face SigLIP layouts.
    """
    siglip_model = model.siglip_model

    if hasattr(siglip_model, 'post_layernorm'):
        return siglip_model.post_layernorm

    if hasattr(siglip_model, 'vision_model') and hasattr(siglip_model.vision_model, 'post_layernorm'):
        return siglip_model.vision_model.post_layernorm

    raise AttributeError(
        f'Could not resolve Grad-CAM target layer from model type: {type(siglip_model).__name__}'
    )


def main():
    parser = argparse.ArgumentParser(description='Load a Python config file.')
    parser.add_argument('--config_path', type=str, default='config/pretrain_classification.py', help='The path to the Python config file')
    parser.add_argument('--checkpoint_path', type=str, default='/content/pretrained_classification.pth', help='The path to the checkpoint file')
    parser.add_argument('--coco_json', type=str, default='/content/SoccerExplainability/annotations-coco.json',
                        help='Path to annotations-coco.json for attribution evaluation against GT bboxes')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default='/content/eval_results.json',
                        help='Optional path to save per-video attribution evaluation results as JSON')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/gradcam-visualizations/',
                        help='Directory to save attribution visualization outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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


    if MODEL_TYPE.lower() == "siglip":
        classifier = SigLIP_Classifier(
            keywords=config_test_dataset['keywords'],
            feature_dim=768,
            model_name="google/siglip-base-patch16-224"
        ).eval()
    else:
        classifier = MatchVision_Classifier(
            keywords=config_test_dataset['keywords'],
            classifier_transformer_type=classifier_transformer_type,
            vision_encoder_type=encoder_type,
            use_transformer=use_transformer,
        ).eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    classifier.load_state_dict(new_state_dict)

    classifier = classifier.to(devices[0])
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)

    print(classifier.module.transformer_encoder.layers[-1])

    attribution_evaluator = None
    all_eval_results = {}
    if args.coco_json and os.path.exists(args.coco_json):
        attribution_evaluator = CocoAttributionEvaluator(args.coco_json)
        print(f"Loaded COCO annotations from {args.coco_json} "
              f"({len(attribution_evaluator.coco['annotations'])} annotations, "
              f"{len(attribution_evaluator.coco['images'])} images)")
    elif args.coco_json:
        print(f'Warning: COCO JSON not found at {args.coco_json}, skipping evaluation.')

    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Inference')

    for _, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:
        video_name = video_path[0].split('/')[-1]
        print(f'Processing video: {video_name}')

        matched_video_id = None
        if attribution_evaluator is not None:
            vp = video_path[0]
            for ann_vid in attribution_evaluator.get_annotated_video_ids():
                if vp.endswith(ann_vid) or ann_vid in vp:
                    matched_video_id = ann_vid
                    break

        frames = frames.to(devices[0])
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)

        logits = classifier.module.forward(frames)

        gradcam_model = classifier.module if hasattr(classifier, 'module') else classifier
        gradcam_target_layer = get_gradcam_target_layer(gradcam_model)
        grad_cam = GradCAM(
            model=gradcam_model,
            target_layers=[gradcam_target_layer],
            reshape_transform=reshape_transform,
        )
        grad_cam_results = grad_cam(input_tensor=frames, targets=[ClassifierOutputTarget(caption[0])])
        print('attribution map batch shape: ', grad_cam_results.shape)

        i = 0
        for grad_cam_result in grad_cam_results:
            new_frames = dummy_frames[i]
            new_frames = new_frames.permute(0, 2, 3, 1)
            print('attribution map shape: ', grad_cam_result.shape)
            grad_cam_mean = torch.mean(torch.tensor(grad_cam_result, device='cpu'), dim=(1, 2)).cpu().numpy()
            print('attribution score shape: ', grad_cam_mean.shape)

            visualizations = []
            for j in range(30):
                visualization = show_cam_on_image(
                    cv2.resize(np.float32(new_frames[j].cpu()) / 255.0, (224, 224)),
                    grad_cam_result[j],
                    use_rgb=True,
                )
                visualizations.append(visualization)

            if attribution_evaluator is not None and matched_video_id is not None:
                eval_result = attribution_evaluator.evaluate_video(
                    grad_cam_result,
                    matched_video_id,
                    start_second=0,
                    cam_threshold=args.cam_threshold,
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

            predictions = classifier.module.get_types(logits)
            print(predictions[i, 0].shape)
            prediction_text = test_dataset.keywords[predictions[i, 0].item()]
            ground_truth_text = caption_text[0]
            print(f'Prediction: {prediction_text}')
            print(f'Ground Truth: {ground_truth_text}')

            save_lowres_visualization_video(
                video_directory=video_directory,
                video_name=video_timestamp,
                lowres_frames=new_frames,
                attribution_maps=grad_cam_result,
                attribution_scores=grad_cam_mean,
                prediction_text=prediction_text,
                ground_truth_text=ground_truth_text,
                attribution_evaluator=attribution_evaluator,
                matched_video_id=matched_video_id if attribution_evaluator else None,
                cam_threshold=args.cam_threshold,
                attribution_method_name='GradCAM',
            )

            i += 1

        all_predictions.append(predictions.cpu())
        del frames

    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    if attribution_evaluator is not None and all_eval_results:
        print('\n===== Attribution Label-Group Evaluation Summary =====')
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
            else:
                print(f'  Note: Video has no annotated frames in any group: {res["video_id"]}')

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

        if args.eval_output_json:
            import json as _json

            def _conv(obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            with open(args.eval_output_json, 'w') as _f:
                _json.dump(
                    {
                        'per_video': {k: v for k, v in all_eval_results.items()},
                        'global_label_group_summary': global_group_summary,
                    },
                    _f,
                    indent=2,
                    default=_conv,
                )
            print(f'  Detailed results saved to {args.eval_output_json}')


if __name__ == '__main__':
    main()
