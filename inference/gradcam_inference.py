import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from config.model_type import MODEL_TYPE
from model.SigLIP_classifier import SigLIP_Classifier
from model.SoccerMaster.SoccerMaster_multi_task import MultiTaskingModel

from inference_utils import (
    load_config, reshape_transform, create_test_dataloader,
    load_classifier, setup_attribution_evaluator, match_video_id,
    evaluate_and_print_video, print_and_save_eval_summary,
)
from visualization_video import save_lowres_visualization_video


base_path = '/content/drive/MyDrive/arsenal-paris-gradcam/'
high_res_video_path = '/content/drive/MyDrive/arsenal-paris-high-res/2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG/'


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
    'CAPTION_CLASSIFICATION_LOSS_WEIGHT': 1.0,
    'CAPTION_CLASSIFICATION_LABEL_SMOOTHING': 0.0,
}


class SoccerMasterClassifierAdapter(nn.Module):
    """Adapter exposing SoccerMaster as a classifier for Grad-CAM."""

    def __init__(self, multitask_model, input_size=512, dataset_name='VideoCaption'):
        super().__init__()
        self.multitask_model = multitask_model
        self.input_size = input_size
        self.dataset_name = dataset_name
        # For Grad-CAM target-layer resolution.
        self.siglip_model = multitask_model.backbone.vision_model

    def forward(self, x):
        # Accept (B, C, T, H, W) and convert to SoccerMaster format (B, T, C, H, W).
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        if x.dim() != 5:
            raise ValueError(f'Unexpected input rank {x.dim()} for SoccerMasterClassifierAdapter')

        bsz, timesteps, channels, height, width = x.shape
        if height != self.input_size or width != self.input_size:
            x = x.reshape(bsz * timesteps, channels, height, width)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            x = x.reshape(bsz, timesteps, channels, self.input_size, self.input_size)

        outputs = self.multitask_model(x, dataset_name=self.dataset_name, metas=None, text=None)
        return outputs['CaptionClassification']['logits']

    def get_types(self, logits):
        _, top_indices = torch.topk(logits, k=5, dim=1, largest=True, sorted=True)
        return top_indices


def load_soccermaster_classifier(checkpoint_dir, device):
    config = SOCCERMASTER_DEFAULT_CONFIG.copy()
    model = MultiTaskingModel(config)
    model.load_checkpoint(checkpoint_dir)
    model = model.half().to(device).eval()
    print(f'SoccerMaster model loaded on {device} (float16)')
    return SoccerMasterClassifierAdapter(model).eval()


def get_gradcam_target_layer(model):
    """Resolve a SigLIP-compatible target layer for Grad-CAM."""
    # Prefer the SoccerMaster encoder layer if available (explicit path requested).
    # `model` here is the `SoccerMasterClassifierAdapter` or its `.module` when wrapped.
    if hasattr(model, 'multitask_model'):
        try:
            layer = model.multitask_model.backbone.vision_model.encoder_blocks[-1].layer_norm1
            print(f'Using explicit Grad-CAM target layer: {type(layer).__name__}')
            return layer
        except Exception:
            # Fall through to legacy resolution if the explicit path is not present.
            pass

    siglip_model = model.siglip_model

    if hasattr(siglip_model, 'post_norm'):
        return siglip_model.post_norm

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
    parser.add_argument('--siglip_temporal_aggregation', type=str, default='mean', choices=['mean', 'max'],
                        help='Temporal aggregation for SigLIP frame embeddings (default: mean)')
    parser.add_argument('--soccermaster_checkpoint_dir', type=str, default=None,
                        help='Path to SoccerMaster checkpoint directory (backbone.pt + CaptionClassification.pt)')
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

    if MODEL_TYPE.lower() == 'soccermaster':
        config_test_dataset = config_test_dataset.copy()
        config_test_dataset['processor_model_name'] = SOCCERMASTER_DEFAULT_CONFIG['CKPT_PATH']

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    test_dataset, test_data_loader = create_test_dataloader(config_test_dataset)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    configured_model_type = MODEL_TYPE.lower()
    if configured_model_type == 'siglip':
        classifier = SigLIP_Classifier(
            keywords=config_test_dataset['keywords'],
            feature_dim=768,
            model_name='google/siglip-base-patch16-224',
            temporal_aggregation=args.siglip_temporal_aggregation,
        ).eval()
        classifier = classifier.to(devices[0])
        classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
        print(
            'Using Hugging Face pretrained SigLIP weights; skipping --checkpoint_path loading. '
            f'Temporal aggregation: {args.siglip_temporal_aggregation}'
        )
    elif configured_model_type == 'soccermaster':
        if not args.soccermaster_checkpoint_dir:
            raise ValueError('MODEL_TYPE is soccermaster but --soccermaster_checkpoint_dir was not provided.')

        classifier = load_soccermaster_classifier(args.soccermaster_checkpoint_dir, devices[0])
        classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
        print(f'Loaded SoccerMaster checkpoint from: {args.soccermaster_checkpoint_dir}')
    else:
        classifier = load_classifier(
            config_test_dataset, classifier_transformer_type, encoder_type,
            use_transformer, checkpoint_path, devices, device_ids,
        )

    if hasattr(classifier.module, 'transformer_encoder'):
        print(classifier.module.transformer_encoder.layers[-1])

    # ----------------------------------------------------------------
    # COCO evaluator (optional)
    # ----------------------------------------------------------------
    attribution_evaluator, all_eval_results = setup_attribution_evaluator(args.coco_json)

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Inference')

    for _, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:
        video_name = video_path[0].split('/')[-1]
        print(f'Processing video: {video_name}')

        matched_video_id = match_video_id(attribution_evaluator, video_path[0])

        frames = frames.to(devices[0])
        if configured_model_type == 'soccermaster':
            frames = frames.half()
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)

        logits = classifier.module.forward(frames)

        gradcam_model = classifier.module if hasattr(classifier, 'module') else classifier
        gradcam_target_layer = get_gradcam_target_layer(gradcam_model)
        
        # For SoccerMaster, use post_norm after encoder blocks (consistent with other SigLIP models)
        if configured_model_type == 'soccermaster':
            # Use the post_norm layer from the vision backbone which normalizes token outputs
            gradcam_target_layer = gradcam_model.multitask_model.backbone.vision_model.post_norm
            # SoccerMaster uses siglip2-large-patch16-512: input_size=512, patch_size=16 => 32x32=1024 tokens
            gradcam_reshape_transform = lambda result: reshape_transform(
                result, height=32, width=32, timesteps=frames.shape[2]
            )
        else:
            gradcam_reshape_transform = lambda result: reshape_transform(result, timesteps=frames.shape[2])

        grad_cam = GradCAM(
            model=gradcam_model,
            target_layers=[gradcam_target_layer],
            reshape_transform=gradcam_reshape_transform,
        )
        grad_cam_results = grad_cam(input_tensor=frames, targets=[ClassifierOutputTarget(caption[0])])
        print('attribution map batch shape: ', grad_cam_results.shape)

        i = 0
        for grad_cam_result in grad_cam_results:
            new_frames = dummy_frames[i]
            new_frames = new_frames.permute(0, 2, 3, 1)
            grad_cam_result = np.asarray(grad_cam_result)

            if grad_cam_result.ndim == 2:
                grad_cam_result = np.repeat(grad_cam_result[None, ...], new_frames.shape[0], axis=0)
            elif grad_cam_result.ndim == 3 and grad_cam_result.shape[0] == 1 and new_frames.shape[0] > 1:
                grad_cam_result = np.repeat(grad_cam_result, new_frames.shape[0], axis=0)

            if grad_cam_result.shape[0] != new_frames.shape[0]:
                raise ValueError(
                    f'Attribution map frame count ({grad_cam_result.shape[0]}) does not match '
                    f'video frame count ({new_frames.shape[0]}).'
                )

            print('attribution map shape: ', grad_cam_result.shape)
            grad_cam_mean = torch.mean(torch.tensor(grad_cam_result, device='cpu'), dim=(1, 2)).cpu().numpy()
            print('attribution score shape: ', grad_cam_mean.shape)

            visualizations = []
            for j in range(new_frames.shape[0]):
                visualization = show_cam_on_image(
                    cv2.resize(np.float32(new_frames[j].cpu()) / 255.0, (224, 224)),
                    grad_cam_result[j],
                    use_rgb=True,
                )
                visualizations.append(visualization)

            # ----------------------------------------------------------
            # COCO evaluation
            # ----------------------------------------------------------
            evaluate_and_print_video(
                attribution_evaluator, grad_cam_result, matched_video_id,
                video_name, args.cam_threshold, all_eval_results,
            )

            # ----------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------
            predictions = classifier.module.get_types(logits)
            print(predictions[i, 0].shape)
            prediction_text = test_dataset.keywords[predictions[i, 0].item()]
            ground_truth_text = caption_text[0]
            print(f'Prediction: {prediction_text}')
            print(f'Ground Truth: {ground_truth_text}')

            # ----------------------------------------------------------
            # Visualization
            # ----------------------------------------------------------
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
        del logits
        del predictions
        del grad_cam_results
        del grad_cam
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    # ----------------------------------------------------------------
    # Global evaluation summary
    # ----------------------------------------------------------------
    if attribution_evaluator is not None and all_eval_results:
        print_and_save_eval_summary(
            all_eval_results,
            eval_output_path=args.eval_output_json,
            summary_title='GradCAM Attribution Label-Group Evaluation Summary',
        )


if __name__ == '__main__':
    main()
