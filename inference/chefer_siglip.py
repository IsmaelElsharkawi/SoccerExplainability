"""
Chefer explainability inference for plain SigLIP classification.

This path uses the Hugging Face SigLIP vision encoder directly on each frame,
then aggregates frame-level pooled embeddings with mean or max before computing
image-text similarity logits against the class-name text embeddings.

The Chefer relevance propagation remains per-frame and spatial-only:
    1. Forward pass through SigLIP vision encoder on each frame
    2. Aggregate frame embeddings exactly as SigLIP_Classifier.forward()
    3. Backward from the target class logit
    4. Propagate relevance through all spatial self-attention layers
    5. Extract per-frame heatmaps via the pooling-head probe cross-attention
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.SigLIP_classifier import SigLIP_Classifier
from inference_utils import (
    load_config,
    create_test_dataloader,
    setup_attribution_evaluator,
    match_video_ids,
    evaluate_and_print_video,
    print_and_save_eval_summary,
    LABEL_NAMES,
    LABEL_TO_IDX,
    wrap_siglip_attention_module,
    wrap_pooling_head_attention,
    chefer_attribution_renderer,
)
from visualization_video import save_lowres_visualization_video


def wrap_siglip_model(model):
    """Wrap SigLIP vision attention modules for Chefer explainability."""
    wrapped = {
        'spatial_attn': [],
        'pooling_head_attn': None,
    }

    vision_model = model.siglip_model.vision_model

    for i, block in enumerate(vision_model.encoder.layers):
        wrapped_spatial = wrap_siglip_attention_module(block.self_attn)
        wrapped['spatial_attn'].append((i, wrapped_spatial))

    wrapped['pooling_head_attn'] = wrap_pooling_head_attention(vision_model.head.attention)

    print('Wrapped attention modules (SigLIP per-frame spatial-only):')
    print(f"  - Spatial (SigLIP): {len(wrapped['spatial_attn'])} layers")
    print('  - Pooling head (probe->patches): yes')

    return wrapped


def _resolve_target_label(cls_logits, target_label, target_label_name):
    if target_label_name is not None:
        normalized_name = target_label_name.lower()
        if normalized_name not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{target_label_name}'. Available: {LABEL_NAMES}")
        target_label = LABEL_TO_IDX[normalized_name]
        print(f"Resolved label name '{target_label_name}' to index {target_label}")

    if target_label is None:
        target_label = int(cls_logits[0].argmax().item())
        print(f'No target label provided. Using top prediction index {target_label}')

    return target_label


def generate_per_frame_heatmaps_siglip(
    model,
    video_frames: torch.Tensor,
    target_label: Optional[int] = None,
    target_label_name: Optional[str] = None,
    device: str = 'cuda',
    num_frames: Optional[int] = None,
    patch_size: int = 14,
) -> np.ndarray:
    """Generate per-frame spatial heatmaps using Chefer on plain SigLIP."""
    model.eval()
    wrapped = wrap_siglip_model(model)

    video_frames = video_frames.to(device)

    if video_frames.dim() == 4:
        video_frames = video_frames.unsqueeze(2)
    elif video_frames.dim() != 5:
        raise ValueError(f'Unexpected input rank {video_frames.dim()} for SigLIP Chefer')

    video_frames.requires_grad_(True)

    batch_size, _, total_frames, _, _ = video_frames.shape
    num_frames = total_frames if num_frames is None else num_frames
    num_patches = patch_size * patch_size

    if num_frames != total_frames:
        raise ValueError(f'num_frames={num_frames} does not match input frames={total_frames}')

    r_pp_per_frame = [torch.eye(num_patches, device=device) for _ in range(num_frames)]

    model.zero_grad()

    frame_inputs = video_frames.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames,
        video_frames.shape[1],
        video_frames.shape[3],
        video_frames.shape[4],
    )

    outputs = model.siglip_model(pixel_values=frame_inputs)
    frame_embeds = outputs.pooler_output.reshape(batch_size, num_frames, -1)
    image_embeds = model._aggregate_temporal_embeddings(frame_embeds)

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = model.text_embeds
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    cls_logits = image_embeds @ text_embeds.t()

    target_label = _resolve_target_label(cls_logits, target_label, target_label_name)
    target = cls_logits[0, target_label]
    print(f"Chefer class-logit backprop: class_idx={target_label}, logit={target.item():.4f}")

    target.backward(retain_graph=True)
    print(f"Computed backward pass. Target value: {target.item():.4f}")

    for _, attn_module in wrapped['spatial_attn']:
        attn_probs = attn_module.get_attn()
        attn_grad = attn_module.get_attn_gradients()

        if attn_probs is None or attn_grad is None:
            continue

        for t in range(num_frames):
            cam = attn_probs[t]
            grad = attn_grad[t]
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            r_pp_per_frame[t] += torch.matmul(cam, r_pp_per_frame[t])

    head_attn_module = wrapped['pooling_head_attn']
    head_attn = head_attn_module.get_attn()
    head_grad = head_attn_module.get_attn_gradients()

    if head_attn is None or head_grad is None:
        raise RuntimeError('SigLIP pooling-head attention gradients were not captured during Chefer backward pass.')

    print(f"  Pooling head: cross-attention {head_attn.shape} + gradients {head_grad.shape}")

    heatmaps = []
    for t in range(num_frames):
        cam = head_attn[t]
        grad = head_grad[t]
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)

        image_relevance = torch.matmul(cam, r_pp_per_frame[t]).squeeze(0)
        heatmap = image_relevance.detach().cpu().numpy().reshape(patch_size, patch_size)
        heatmaps.append(heatmap)

    heatmaps = np.stack(heatmaps, axis=0)

    for t in range(num_frames):
        heatmap = heatmaps[t]
        if heatmap.max() > heatmap.min():
            heatmaps[t] = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmaps[t] = np.zeros_like(heatmap)

    return heatmaps


def main():
    parser = argparse.ArgumentParser(
        description='Chefer explainability inference (per-frame spatial-only) for plain SigLIP classification.'
    )
    parser.add_argument('--config_path', type=str,
                        default='config/pretrain_classification_cluster.py',
                        help='Path to the Python config file')
    parser.add_argument('--model_name', type=str,
                        default='google/siglip-base-patch16-224',
                        help='Hugging Face model ID or local path for SigLIP vision/text encoders')
    parser.add_argument('--coco_json', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'annotations-coco.json'),
                        help='Path to annotations-coco.json for attribution evaluation')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default=None,
                        help='Optional path to save per-video evaluation results as JSON')
    parser.add_argument('--output_dir', type=str,
                        default='../output_chefer_siglip/',
                        help='Directory to save attribution visualization outputs')
    parser.add_argument('--siglip_temporal_aggregation', type=str, default='mean', choices=['mean', 'max'],
                        help='Temporal aggregation for SigLIP frame embeddings (default: mean)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config_path)

    config_dataset = config['dataset']
    config_test_dataset = config_dataset['test']

    config_training_settings = config['training_settings']
    device_ids = config_training_settings['device_ids']
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    test_dataset, test_data_loader = create_test_dataloader(config_test_dataset)

    classifier = SigLIP_Classifier(
        keywords=config_test_dataset['keywords'],
        feature_dim=768,
        model_name=args.model_name,
        temporal_aggregation=args.siglip_temporal_aggregation,
    ).eval()
    classifier = classifier.to(devices[0])
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)

    print(
        'Using Hugging Face pretrained SigLIP weights for Chefer inference. '
        f'Temporal aggregation: {args.siglip_temporal_aggregation}'
    )
    print(f"Model loaded. Keywords ({len(test_dataset.keywords)}): {test_dataset.keywords}")

    coco_json_path = os.path.abspath(args.coco_json)
    attribution_evaluator, all_eval_results = setup_attribution_evaluator(coco_json_path)

    all_predictions = []
    test_progress_bar = tqdm(
        enumerate(test_data_loader),
        total=len(test_data_loader),
        desc='Chefer SigLIP Inference',
    )

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

        target_label_idx = caption[0].item()
        chefer_heatmaps = generate_per_frame_heatmaps_siglip(
            classifier.module,
            frames,
            target_label=target_label_idx,
            device=str(devices[0]),
            num_frames=frames.shape[2],
        )
        print(f'Chefer SigLIP heatmap shape: {chefer_heatmaps.shape}')

        chefer_scores = chefer_heatmaps.mean(axis=(1, 2))
        print(f'Chefer scores shape: {chefer_scores.shape}')

        with torch.no_grad():
            logits = classifier.module.forward(frames)

        predictions = classifier.module.get_types(logits)

        for sample_idx in range(frames.shape[0]):
            new_frames = dummy_frames[sample_idx]
            new_frames = new_frames.permute(0, 2, 3, 1)

            evaluate_and_print_video(
                attribution_evaluator,
                chefer_heatmaps,
                matched_video_ids,
                video_name,
                args.cam_threshold,
                all_eval_results,
            )

            prediction_text = test_dataset.keywords[predictions[sample_idx, 0].item()]
            ground_truth_text = caption_text[sample_idx]
            print(f'Prediction: {prediction_text}')
            print(f'Ground Truth: {ground_truth_text}')

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
                attribution_method_name='Chefer-SigLIP',
                attribution_renderer=chefer_attribution_renderer,
            )

        all_predictions.append(predictions.cpu())
        del frames

    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    if attribution_evaluator is not None and all_eval_results:
        eval_output = args.eval_output_json
        if eval_output is None:
            eval_output = os.path.join(args.output_dir, 'chefer_siglip_eval_results.json')
        print_and_save_eval_summary(
            all_eval_results,
            eval_output_path=eval_output,
            summary_title='Chefer SigLIP Attribution Label-Group Evaluation Summary',
        )


if __name__ == '__main__':
    main()