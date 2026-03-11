import sys, os
sys.path.append('/content/UniSoccer')
from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from model.MatchVision_classifier import MatchVision_Classifier
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import importlib.util
import cv2
import numpy as np

from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import glob
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# COCO GradCAM Evaluation – inlined helpers & evaluator
# ---------------------------------------------------------------------------

def _normalised_bbox_to_mask(bbox, H, W):
    """Convert a normalised [x, y, w, h] bbox to a binary mask (H, W)."""
    x, y, w, h = bbox
    x1, y1 = max(0, int(round(x * W))), max(0, int(round(y * H)))
    x2, y2 = min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H)))
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _combine_bbox_masks(bboxes, H, W):
    """Union of multiple normalised bbox masks."""
    combined = np.zeros((H, W), dtype=np.float32)
    for bbox in bboxes:
        combined = np.maximum(combined, _normalised_bbox_to_mask(bbox, H, W))
    return combined


def _compute_heatmap_metrics_for_bboxes(gradcam_heatmap, bboxes, cam_threshold=0.5):
    """Compute Energy / Pointing / IoU for a given bbox subset."""
    H, W = gradcam_heatmap.shape[:2]
    gt_mask = _combine_bbox_masks(bboxes, H, W)

    total_energy = gradcam_heatmap.sum()
    if total_energy > 0:
        energy_inside_bbox = float((gradcam_heatmap * gt_mask).sum() / total_energy)
    else:
        energy_inside_bbox = 0.0

    peak_idx = np.unravel_index(np.argmax(gradcam_heatmap), gradcam_heatmap.shape)
    pointing_accuracy = float(gt_mask[peak_idx] > 0)

    cam_binary = (gradcam_heatmap >= cam_threshold * gradcam_heatmap.max()).astype(np.float32)
    intersection = (cam_binary * gt_mask).sum()
    union = np.clip(cam_binary + gt_mask, 0, 1).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    return {
        "energy_inside_bbox": energy_inside_bbox,
        "pointing_accuracy": pointing_accuracy,
        "iou": iou,
    }


def _annotation_display_name(annotation, categories):
    """Resolve the label text shown for an annotation."""
    return (
        annotation.get("label")
        or categories.get(annotation.get("category_id"), "unknown")
    )


def _annotation_type_key(annotation, categories):
    """Stable type key used for color assignment."""
    return categories.get(annotation.get("category_id"), annotation.get("label", "unknown"))


def _color_for_type(type_key):
    """Return a stable BGR color for each annotation type."""
    palette = [
        (0, 255, 255),
        (255, 140, 0),
        (0, 220, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 0),
        (180, 105, 255),
        (255, 80, 80),
    ]
    return palette[sum(ord(char) for char in str(type_key)) % len(palette)]


def draw_gt_bboxes_on_frame(frame, annotations, categories, thickness=2):
    """Draw normalized bounding boxes and labels on an image (copy returned)."""
    H, W = frame.shape[:2]
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1

    for annotation in annotations:
        bbox = annotation["bbox"]
        x, y, w, h = bbox
        x1, y1 = int(round(x * W)), int(round(y * H))
        x2, y2 = int(round((x + w) * W)), int(round((y + h) * H))
        color = _color_for_type(_annotation_type_key(annotation, categories))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label_text = _annotation_display_name(annotation, categories)
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )
        text_x = max(0, min(x1, W - text_w - 6))

        if y1 - text_h - baseline - 8 >= 0:
            text_y = y1 - 6
            box_top = text_y - text_h - baseline - 2
            box_bottom = text_y + 2
        else:
            text_y = min(H - baseline - 2, y1 + text_h + baseline + 6)
            box_top = text_y - text_h - baseline - 2
            box_bottom = text_y + 2

        box_top = max(0, box_top)
        box_bottom = min(H - 1, box_bottom)
        box_right = min(W - 1, text_x + text_w + 6)

        cv2.rectangle(
            out,
            (text_x, box_top),
            (box_right, box_bottom),
            color,
            -1,
        )
        cv2.putText(
            out,
            label_text,
            (text_x + 3, text_y),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    return out


def overlay_metrics_text(frame, metrics, position=(10, 25)):
    """Overlay evaluation metrics as text on the frame."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = position[1]
    for key in ["energy_inside_bbox", "pointing_accuracy", "iou"]:
        val = metrics.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            text = f"{key}: {val:.3f}"
            cv2.putText(out, text, (position[0], y), font, 0.5, (255, 255, 255), 1)
            y += 20
    return out


class CocoGradCamEvaluator:
    """Load COCO annotations and evaluate GradCAM heatmaps against them."""

    def __init__(self, coco_json_path):
        with open(coco_json_path, "r") as f:
            self.coco = json.load(f)

        # Store original images without modifying them
        self.images_by_id = {img["id"]: img.copy() for img in self.coco["images"]}

        self.image_id_by_video_second = {
            (img["video_id"], img["second"]): img["id"]
            for img in self.coco["images"]
        }

        # Resolve image dimensions: fix placeholder 1×1 entries by
        # finding the correct dimensions from images with the same video_id.
        # Build a mapping of video_id to valid dimensions (first valid entry found).
        video_dims = {}
        for img in self.coco["images"]:
            vid = img["video_id"]
            if vid not in video_dims and img["width"] > 1 and img["height"] > 1:
                video_dims[vid] = (img["width"], img["height"])

        # Fix placeholder dimensions in our local copy only
        # Only fix for videos where we found valid dimensions in the first pass
        for img_id, img in self.images_by_id.items():
            if img["width"] <= 1 or img["height"] <= 1:
                vid = img["video_id"]
                # Only apply fix if we found valid dimensions for this video
                if vid in video_dims:
                    img["width"], img["height"] = video_dims[vid]
                # Otherwise leave as placeholder - these won't be normalized

        # Normalise absolute-pixel bboxes to 0-1 range using image dimensions
        # Store normalized copy of annotations without modifying originals
        self.annots_by_image = defaultdict(list)
        for ann in self.coco["annotations"]:
            img = self.images_by_id.get(ann["image_id"])
            if img:
                # Create a normalized copy of the annotation
                ann_copy = ann.copy()
                # Only normalize if image has valid dimensions (not placeholder 1x1)
                # and annotation is marked as having ROI
                if not ann.get("no_roi", False) and img["width"] > 1 and img["height"] > 1:
                    iw, ih = img["width"], img["height"]
                    bx, by, bw, bh = ann["bbox"]
                    # Check if bbox is already in normalized range (0-1)
                    if bx <= 1.0 and by <= 1.0 and (bx + bw) <= 1.0 and (by + bh) <= 1.0:
                        # Bbox is already normalized, keep as-is
                        pass
                    else:
                        # Bbox is in pixel coordinates, normalize to 0-1 range
                        ann_copy["bbox"] = [bx / iw, by / ih, bw / iw, bh / ih]
                self.annots_by_image[ann["image_id"]].append(ann_copy)

        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco.get("categories", [])
        }

    # ------------------------------------------------------------------
    def get_annotations_for_frame(self, video_id, second):
        image_id = self.image_id_by_video_second.get((video_id, second))
        if image_id is None:
            return None
        annots = self.annots_by_image.get(image_id)
        return annots if annots else None

    def has_annotation(self, video_id, second):
        return (video_id, second) in self.image_id_by_video_second

    def evaluate_frame(self, gradcam_heatmap, video_id, second,
                       iou_threshold=0.5, cam_threshold=0.5):
        """
        Evaluate a single GradCAM heatmap (H, W) against ground-truth bboxes.

        Returns dict with metrics, or None if no annotation exists.
        """
        annots = self.get_annotations_for_frame(video_id, second)
        if annots is None:
            return None
        
        # Ismael: This shape might not be accurate

        H, W = gradcam_heatmap.shape[:2]
        roi_annots = [a for a in annots if not a.get("no_roi", False)]
        is_no_roi = len(roi_annots) == 0
        metrics = {"has_roi": 0.0 if is_no_roi else 1.0}

        if is_no_roi:
            flat = gradcam_heatmap.flatten()
            total = flat.sum()
            if total > 0:
                p = flat / total
                p = p[p > 0]
                entropy = -np.sum(p * np.log(p))
                max_entropy = np.log(len(flat))
                metrics["normalised_entropy"] = float(entropy / max_entropy)
            else:
                metrics["normalised_entropy"] = 1.0
            metrics["energy_inside_bbox"] = float("nan")
            metrics["pointing_accuracy"] = float("nan")
            metrics["iou"] = float("nan")
            return metrics

        bboxes = [a["bbox"] for a in roi_annots]
        overall_scores = _compute_heatmap_metrics_for_bboxes(
            gradcam_heatmap, bboxes, cam_threshold=cam_threshold
        )
        metrics.update(overall_scores)

        # 4. Per-annotation breakdown
        per_ann = []
        total_energy = gradcam_heatmap.sum()
        cam_binary = (
            gradcam_heatmap >= cam_threshold * gradcam_heatmap.max()
        ).astype(np.float32)
        peak_idx = np.unravel_index(
            np.argmax(gradcam_heatmap), gradcam_heatmap.shape
        )
        for ann in roi_annots:
            ann_mask = _normalised_bbox_to_mask(ann["bbox"], H, W)
            ann_energy = (gradcam_heatmap * ann_mask).sum()
            ann_pointing = float(ann_mask[peak_idx] > 0)
            ann_inter = (cam_binary * ann_mask).sum()
            ann_union = np.clip(cam_binary + ann_mask, 0, 1).sum()
            per_ann.append({
                "annotation_id": ann["id"],
                "category": self.categories.get(ann.get("category_id"), "unknown"),
                "label": ann.get("label", ""),
                "energy": float(ann_energy / total_energy) if total_energy > 0 else 0.0,
                "pointing": ann_pointing,
                "iou": float(ann_inter / ann_union) if ann_union > 0 else 0.0,
            })
        metrics["per_annotation"] = per_ann

        # 5. Scores by cumulative label groups requested by annotation policy
        #    a) small label only
        #    b) small + large labels
        #    c) small + large labels + visual cues
        def _norm_label(a):
            return str(a.get("label", "")).strip().lower()

        group_to_labels = {
            "small_only": {"small label"},
            "small_large": {"small label", "large label"},
            "small_large_visual_cues": {"small label", "large label", "visual cue"},
        }
        group_scores = {}
        for group_name, allowed_labels in group_to_labels.items():
            group_bboxes = [a["bbox"] for a in roi_annots if _norm_label(a) in allowed_labels]
            if group_bboxes:
                group_scores[group_name] = _compute_heatmap_metrics_for_bboxes(
                    gradcam_heatmap, group_bboxes, cam_threshold=cam_threshold
                )
                group_scores[group_name]["frames_with_group"] = 1
            else:
                group_scores[group_name] = {
                    "energy_inside_bbox": float("nan"),
                    "pointing_accuracy": float("nan"),
                    "iou": float("nan"),
                    "frames_with_group": 0,
                }
        metrics["label_group_scores"] = group_scores
        return metrics

    def evaluate_video(self, gradcam_heatmaps, video_id,
                       start_second=0, iou_threshold=0.5, cam_threshold=0.5):
        """
        Evaluate all T GradCAM frames for a video clip.

        Args:
            gradcam_heatmaps: (T, H, W) array of heatmaps.
            video_id: matches images[].video_id in COCO JSON.
        Returns:
            dict with per_frame metrics and aggregated summary.
        """
        T = gradcam_heatmaps.shape[0]
        per_frame = []
        energies, pointings, ious = [], [], []
        group_acc = {
            "small_only": {"energy": [], "pointing": [], "iou": []},
            "small_large": {"energy": [], "pointing": [], "iou": []},
            "small_large_visual_cues": {"energy": [], "pointing": [], "iou": []},
        }

        for t in range(T):
            sec = start_second + t
            fm = self.evaluate_frame(
                gradcam_heatmaps[t], video_id, sec,
                iou_threshold=iou_threshold, cam_threshold=cam_threshold,
            )
            per_frame.append({"second": sec, "metrics": fm})
            if fm is not None and fm["has_roi"]:
                energies.append(fm["energy_inside_bbox"])
                pointings.append(fm["pointing_accuracy"])
                ious.append(fm["iou"])
                for group_name, group_metrics in fm.get("label_group_scores", {}).items():
                    if not np.isnan(group_metrics.get("energy_inside_bbox", np.nan)):
                        group_acc[group_name]["energy"].append(group_metrics["energy_inside_bbox"])
                    if not np.isnan(group_metrics.get("pointing_accuracy", np.nan)):
                        group_acc[group_name]["pointing"].append(group_metrics["pointing_accuracy"])
                    if not np.isnan(group_metrics.get("iou", np.nan)):
                        group_acc[group_name]["iou"].append(group_metrics["iou"])

        summary = {}
        if energies:
            summary["mean_energy_inside_bbox"] = float(np.mean(energies))
            summary["mean_pointing_accuracy"] = float(np.mean(pointings))
            summary["mean_iou"] = float(np.mean(ious))
        else:
            summary["mean_energy_inside_bbox"] = float("nan")
            summary["mean_pointing_accuracy"] = float("nan")
            summary["mean_iou"] = float("nan")
        summary["annotated_frames"] = len(energies)
        summary["total_frames"] = T
        summary["label_group_scores"] = {}
        for group_name, vals in group_acc.items():
            if vals["energy"]:
                summary["label_group_scores"][group_name] = {
                    "mean_energy_inside_bbox": float(np.mean(vals["energy"])),
                    "mean_pointing_accuracy": float(np.mean(vals["pointing"])),
                    "mean_iou": float(np.mean(vals["iou"])),
                    "annotated_frames": len(vals["energy"]),
                }
            else:
                summary["label_group_scores"][group_name] = {
                    "mean_energy_inside_bbox": float("nan"),
                    "mean_pointing_accuracy": float("nan"),
                    "mean_iou": float("nan"),
                    "annotated_frames": 0,
                }
        return {"per_frame": per_frame, "summary": summary}

    def get_annotated_video_ids(self):
        return sorted(set(img["video_id"] for img in self.coco["images"]))


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config

def reshape_transform(result, height=14, width=14, timesteps=30):
    # Result shape: [batch* time, 196, embedding_dim]
    BT, N, C = result.shape 
    print("Tensor Shape:", result.shape)
    result = result.unsqueeze(0)  # Add batch dimension if not present
    result = result.reshape(BT//timesteps, timesteps, height , width, C)
    
    # 3. Transpose dimensions to get [batch*time, embedding_dim, height, width]
    result = result.permute(0, 4, 1,2,3) # Add depth dimension for 3D CAM
    print("Reshaped Tensor Shape:", result.shape)
    return result

# **** Inference Debugging Output ****
# Inference:   0% 0/1 [00:00<?, ?it/s]tensor([16])
# torch.Size([1, 30, 3, 224, 398])
# input shape:  torch.Size([1, 3, 30, 224, 224])
# torch.Size([1, 30, 768])
# Tensor Shape: torch.Size([30, 196, 768])
# Reshaped Tensor Shape: torch.Size([1, 768, 30, 14, 14])
# torch.Size([1, 30, 768])
# Tensor Shape: torch.Size([30, 196, 768])
# Reshaped Tensor Shape: torch.Size([1, 768, 30, 14, 14])
# inners of gradcam:  (1, 768, 30, 14, 14)
# grad cam results shape:  (1, 30, 224, 224)
# grad cam result shape:  (30, 224, 224)
# visualization shape:  (224, 224, 3)
# Inference: 100% 1/1 [00:06<00:00,  6.09s/it]
# tensor([[16, 11,  2, 13,  5]])

base_path = '/content/drive/MyDrive/arsenal-paris-gradcam/'

high_res_video_path = '/content/drive/MyDrive/arsenal-paris-high-res/2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG/'

def create_combined_visualization(high_res_frame, gradcam_heatmap, gradcam_graph, 
                                   prediction_text, ground_truth_text, frame_number):
    """
    Create a combined visualization with high-res video, GradCAM heatmap, graph, 
    inference result, and ground truth label.
    
    Args:
        high_res_frame: High resolution video frame (H, W, 3)
        gradcam_heatmap: GradCAM heatmap visualization (224, 224, 3)
        gradcam_graph: Matplotlib figure for GradCAM scores
        prediction_text: Predicted class name
        ground_truth_text: Ground truth class name
        frame_number: Current frame number
    
    Returns:
        Combined visualization frame
    """
    # Get original dimensions
    hr_h, hr_w = high_res_frame.shape[:2]
    
    # Resize GradCAM heatmap to make it larger (from 224 to 448)
    gradcam_heatmap_resized = cv2.resize(gradcam_heatmap, (448, 448))
    
    # Convert matplotlib figure to image
    canvas = FigureCanvasAgg(gradcam_graph)
    canvas.draw()
    graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (4,))
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)
    
    # Resize graph to match heatmap width (448) while maintaining aspect ratio
    graph_h, graph_w = graph_image.shape[:2]
    new_graph_width = 448
    new_graph_height = int(graph_h * new_graph_width / graph_w)
    graph_resized = cv2.resize(graph_image, (new_graph_width, new_graph_height))
    
    # Stack heatmap and graph vertically
    right_column = np.vstack([gradcam_heatmap_resized, graph_resized])
    
    # Add padding to right column if needed to match high-res frame height
    right_h = right_column.shape[0]
    if right_h < hr_h:
        padding = np.ones((hr_h - right_h, right_column.shape[1], 3), dtype=np.uint8) * 255
        right_column = np.vstack([right_column, padding])
    elif right_h > hr_h:
        right_column = right_column[:hr_h, :, :]
    
    # Create text overlay area
    text_height = 80
    text_area = np.ones((text_height, hr_w + right_column.shape[1], 3), dtype=np.uint8) * 255
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Frame number
    cv2.putText(text_area, f"Frame: {frame_number}/30", (10, 25), 
                font, font_scale, (0, 0, 0), thickness)
    
    # Ground truth (green if match, red if not)
    gt_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
    cv2.putText(text_area, f"Ground Truth: {ground_truth_text}", (10, 50), 
                font, font_scale, gt_color, thickness)
    
    # Prediction
    pred_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
    cv2.putText(text_area, f"Prediction: {prediction_text}", (10, 75), 
                font, font_scale, pred_color, thickness)
    
    # Combine everything horizontally (high-res + right column)
    combined_main = np.hstack([high_res_frame, right_column])
    
    # Stack text area on top
    final_frame = np.vstack([text_area, combined_main])
    
    return final_frame


def save_combined_video(video_directory, video_name, high_res_video_file, 
                        gradcam_scores, prediction_text, ground_truth_text,
                        keywords, visualizations):
    """
    Create and save a combined visualization video.
    
    Args:
        video_directory: Directory containing saved GradCAM frames
        video_name: Name of the video
        high_res_video_file: Path to high resolution video file
        gradcam_scores: Array of GradCAM scores per frame
        prediction_text: Predicted class name
        ground_truth_text: Ground truth class name
        keywords: List of all class keywords
        visualizations: List of GradCAM visualization images (30 frames)
    """
    # Open high-res video
    cap = cv2.VideoCapture(high_res_video_file)
    if not cap.isOpened():
        print(f"Error: Could not open high-res video {high_res_video_file}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We have 30 gradcam frames sampled at 1fps, calculate duration
    duration_seconds = 30  # 30 seconds total (30 frames at 1fps)
    frames_per_gradcam = int(fps)  # Number of video frames per GradCAM frame (1 second worth)
    
    output_path = os.path.join(video_directory, f'{video_name}_combined_visualization.mp4')
    
    # We'll set the output size after reading the first frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    # Pre-generate all GradCAM graphs to avoid recreating them
    gradcam_graphs = []
    for idx in range(30):
        plt.figure(figsize=(6, 3))
        frame_numbers = np.arange(len(gradcam_scores))
        plt.plot(frame_numbers, gradcam_scores, marker='o', linewidth=2, markersize=4, color='blue')
        # Highlight current frame
        plt.plot(idx, gradcam_scores[idx], marker='o', markersize=10, color='red', zorder=5)
        plt.xlabel('Frame Number')
        plt.ylabel('GradCAM Score')
        plt.title(f'GradCAM Scores - {ground_truth_text}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to image
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (4,))
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)
        gradcam_graphs.append(graph_image)
        plt.close()
    
    frame_count = 0
    current_gradcam_idx = 0
    
    # Read and process all frames continuously
    while True:
        ret, high_res_frame = cap.read()
        if not ret:
            break
        
        # Convert from BGR to RGB (OpenCV reads in BGR)
        high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
        
        # Determine which GradCAM visualization to use (updates every second)
        current_gradcam_idx = min(frame_count // frames_per_gradcam, 29)
        
        # Stop after processing 30 seconds worth of frames
        if current_gradcam_idx >= 30:
            break
        
        # Get current GradCAM heatmap and graph
        gradcam_heatmap = visualizations[current_gradcam_idx]
        gradcam_graph_image = gradcam_graphs[current_gradcam_idx]
        
        # Resize heatmap to make it larger (from 224 to 448)
        gradcam_heatmap_resized = cv2.resize(gradcam_heatmap, (448, 448))
        
        # Create combined frame (we need to pass graph as image, not figure)
        # Resize graph to match heatmap width (448) while maintaining aspect ratio
        graph_h, graph_w = gradcam_graph_image.shape[:2]
        new_graph_width = 448
        new_graph_height = int(graph_h * new_graph_width / graph_w)
        graph_resized = cv2.resize(gradcam_graph_image, (new_graph_width, new_graph_height))
        
        # Build visualization manually
        hr_h, hr_w = high_res_frame.shape[:2]
        
        # Stack heatmap and graph vertically
        right_column = np.vstack([gradcam_heatmap_resized, graph_resized])
        
        # Add padding to right column if needed
        right_h = right_column.shape[0]
        if right_h < hr_h:
            padding = np.ones((hr_h - right_h, right_column.shape[1], 3), dtype=np.uint8) * 255
            right_column = np.vstack([right_column, padding])
        elif right_h > hr_h:
            right_column = right_column[:hr_h, :, :]
        
        # Create text overlay area
        text_height = 80
        text_area = np.ones((text_height, hr_w + right_column.shape[1], 3), dtype=np.uint8) * 255
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Frame number
        cv2.putText(text_area, f"Frame: {current_gradcam_idx + 1}/30", (10, 25), 
                    font, font_scale, (0, 0, 0), thickness)
        
        # Ground truth (green if match, red if not)
        gt_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
        cv2.putText(text_area, f"Ground Truth: {ground_truth_text}", (10, 50), 
                    font, font_scale, gt_color, thickness)
        
        # Prediction
        pred_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
        cv2.putText(text_area, f"Prediction: {prediction_text}", (10, 75), 
                    font, font_scale, pred_color, thickness)
        
        # Combine everything horizontally (high-res + right column)
        combined_main = np.hstack([high_res_frame, right_column])
        
        # Stack text area on top
        combined_frame = np.vstack([text_area, combined_main])
        
        # Initialize video writer with first frame dimensions
        if out is None:
            height, width = combined_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)
        
        frame_count += 1
    
    cap.release()
    if out is not None:
        out.release()
        print(f"Combined visualization saved to: {output_path}")
    else:
        print(f"Error: Could not create output video")


def save_lowres_visualization_video(video_directory, video_name, lowres_frames,
                                    gradcam_heatmaps, gradcam_scores,
                                    prediction_text, ground_truth_text,
                                    gradcam_evaluator=None, matched_video_id=None,
                                    cam_threshold=0.5):
    """
    Create and save a visualization video using low-quality dataset frames.

    Each output frame is laid out as:
        ┌─────────────────────────────────────────┐
        │  Frame N/30 | GT: ... | Pred: ...       │
        ├─────────────────────────────────────────┤
        │  GradCAM overlay + black ROI bboxes     │
        ├─────────────────────────────────────────┤
        │          GradCAM score graph            │
        └─────────────────────────────────────────┘

    Args:
        video_directory:    Output directory for the video file.
        video_name:         Base name (without .mp4) for the output file.
        lowres_frames:      (30, H, W, C) uint8 numpy array or tensor of raw frames.
        gradcam_heatmaps:   (30, 224, 224) numpy array of GradCAM heatmaps.
        gradcam_scores:     (30,) numpy array – mean GradCAM activation per frame.
        prediction_text:    Predicted class name.
        ground_truth_text:  Ground-truth class name.
        gradcam_evaluator:  Optional CocoGradCamEvaluator instance.
        matched_video_id:   Video ID inside the COCO annotations.
        cam_threshold:      Threshold for IoU binarisation.
    """
    num_frames = lowres_frames.shape[0]
    target_size = 448  # upscale low-res frames for better visibility

    output_path = os.path.join(video_directory,
                               f'{video_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for idx in range(num_frames):
        # --- prepare frame with GradCAM overlay ---
        frame_raw = lowres_frames[idx]
        if hasattr(frame_raw, 'cpu'):
            frame_raw = frame_raw.cpu().numpy()
        frame_raw = frame_raw.astype(np.uint8)

        frame_float = np.float32(cv2.resize(frame_raw, (224, 224))) / 255.0
        gradcam_vis = show_cam_on_image(frame_float, gradcam_heatmaps[idx],
                                        use_rgb=True)
        gradcam_vis_resized = cv2.resize(gradcam_vis, (target_size, target_size))

        # --- draw ROI bounding boxes (white outline + black inner for contrast) ---
        if gradcam_evaluator is not None and matched_video_id is not None:
            annots = gradcam_evaluator.get_annotations_for_frame(
                matched_video_id, idx)
            if annots:
                roi_annots = [a for a in annots if not a.get('no_roi', False)]
                if roi_annots:
                    # Per-type colored boxes with labels.
                    gradcam_vis_resized = draw_gt_bboxes_on_frame(
                        gradcam_vis_resized,
                        roi_annots,
                        gradcam_evaluator.categories,
                        thickness=3,
                    )

                    # add per-frame metrics text
                    frame_metrics = gradcam_evaluator.evaluate_frame(
                        gradcam_heatmaps[idx], matched_video_id, idx,
                        cam_threshold=cam_threshold,
                    )
                    if frame_metrics:
                        gradcam_vis_resized = overlay_metrics_text(
                            gradcam_vis_resized, frame_metrics)

        panel_width = target_size

        # --- GradCAM score graph ---
        fig, ax = plt.subplots(figsize=(panel_width / 100, 2.0))
        ax.plot(np.arange(num_frames), gradcam_scores,
                marker='o', linewidth=2, markersize=4, color='blue')
        ax.plot(idx, gradcam_scores[idx],
                marker='o', markersize=10, color='red', zorder=5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('GradCAM')
        ax.set_title(f'GradCAM Scores – {ground_truth_text}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (4,))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        graph_img = cv2.resize(graph_img, (panel_width, graph_img.shape[0]))

        # --- text bar ---
        text_height = 80
        text_bar = np.ones((text_height, panel_width, 3), dtype=np.uint8) * 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        match = prediction_text == ground_truth_text
        color = (0, 220, 0) if match else (0, 0, 220)
        cv2.putText(text_bar, f"Frame {idx + 1}/{num_frames}",
                    (10, 25), font, 0.55, (255, 255, 255), 1)
        cv2.putText(text_bar, f"GT: {ground_truth_text}",
                    (10, 50), font, 0.55, color, 2)
        cv2.putText(text_bar, f"Pred: {prediction_text}",
                    (10, 72), font, 0.55, color, 2)

        # --- stack everything vertically ---
        combined = np.vstack([text_bar, gradcam_vis_resized, graph_img])

        if out is None:
            h, w = combined.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 1, (w, h))  # 1 fps

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if out is not None:
        out.release()
        print(f"Low-res visualization saved to: {output_path}")
    else:
        print("Error: Could not create low-res visualization video.")


def main():
    ############## Configs ################
    parser = argparse.ArgumentParser(description="Load a Python config file.")
    parser.add_argument('--config_path', type=str, default="config/pretrain_classification.py", help='The path to the Python config file')
    parser.add_argument('--checkpoint_path', type=str, default="/content/pretrained_classification.pth", help='The path to the checkpoint file')
    parser.add_argument('--coco_json', type=str, default="/content/UniSoccer/annotations-coco.json",
                        help='Path to annotations-coco.json for GradCAM evaluation against GT bboxes')
    parser.add_argument('--cam_threshold', type=float, default=0.5,
                        help='Fraction of max to binarise heatmap for IoU (default: 0.5)')
    parser.add_argument('--eval_output_json', type=str, default="/content/eval_results.json",
                        help='Optional path to save per-video GradCAM evaluation results as JSON')
    parser.add_argument('--output_dir', type=str, default="/content/drive/MyDrive/gradcam-visualizations/",
                        help='Directory to save GradCAM visualization outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config_path)
    checkpoint_path = args.checkpoint_path

    # Dataset configuration
    config_dataset = config["dataset"]
    config_test_dataset = config_dataset["test"]

    # Model configuration
    config_training_settings = config["training_settings"]
    device_ids = config_training_settings["device_ids"]
    classifier_transformer_type = config_training_settings["classifier_transformer_type"]
    encoder_type = config_training_settings["encoder_type"]
    use_transformer = config_training_settings["use_transformer"]

    # Set up the device
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    ############## Dataset ################
    test_dataset_type = None
    if config_test_dataset["balanced_or_not"] == "balanced":
        test_dataset_type = VideoCaptionDataset_Balanced
    else:
        test_dataset_type = VideoCaptionDataset

    test_dataset = test_dataset_type(
        json_file=config_test_dataset["json"],
        video_base_dir=config_test_dataset["video_base"],
        sample=config_test_dataset["sample"],
        keywords=config_test_dataset["keywords"],
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_test_dataset["batch_size"],
        num_workers=config_test_dataset["num_workers"],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    ############## Model ################
    classifier = MatchVision_Classifier(
        keywords=config_test_dataset["keywords"],
        classifier_transformer_type=classifier_transformer_type,
        vision_encoder_type=encoder_type,
        use_transformer=use_transformer
    ).eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    classifier.load_state_dict(new_state_dict)

    # Move model to device and wrap with DataParallel
    classifier = classifier.to(devices[0])
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)

    print(classifier.module.transformer_encoder.layers[-1])

    ############## COCO Evaluation Setup ################
    gradcam_evaluator = None
    all_eval_results = {}  # video_id -> evaluation result
    if args.coco_json and os.path.exists(args.coco_json):
        gradcam_evaluator = CocoGradCamEvaluator(args.coco_json)
        print(f"Loaded COCO annotations from {args.coco_json} "
              f"({len(gradcam_evaluator.coco['annotations'])} annotations, "
              f"{len(gradcam_evaluator.coco['images'])} images)")
    elif args.coco_json:
        print(f"Warning: COCO JSON not found at {args.coco_json}, skipping evaluation.")

    ############## Inference ################
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Inference")

    for batch_idx, (frames, caption, dummy_frames, video_path, caption_text) in test_progress_bar:  # Captions not needed for inference
        video_name = video_path[0].split('/')[-1]
        print(f"Processing video: {video_name}")

        # Try to match video against COCO annotations
        matched_video_id = None
        if gradcam_evaluator is not None:
            vp = video_path[0]
            for ann_vid in gradcam_evaluator.get_annotated_video_ids():
                if vp.endswith(ann_vid) or ann_vid in vp:
                    matched_video_id = ann_vid
                    break

        frames = frames.to(devices[0])
        # Build output path: output_dir / match_name / timestamp.mp4
        vp_parts = video_path[0].replace('\\', '/').split('/')
        match_name = vp_parts[-2] if len(vp_parts) >= 2 else 'unknown_match'
        video_timestamp = video_name.replace('.mp4', '')
        video_directory = os.path.join(args.output_dir, match_name)
        os.makedirs(video_directory, exist_ok=True)
        
        # Forward pass
        logits = classifier.module.forward(frames)

        grad_cam = GradCAM(model=classifier.module, target_layers=[classifier.module.siglip_model.post_layernorm], reshape_transform=reshape_transform)
        grad_cam_results = grad_cam(input_tensor=frames, targets=[ClassifierOutputTarget(caption[0])])
        print("grad cam results shape: ", grad_cam_results.shape)
        i = 0
        for grad_cam_result in grad_cam_results:
            new_frames = dummy_frames[i]
            new_frames = new_frames.permute(0,2,3,1)
            print("grad cam result shape: ", grad_cam_result.shape)
            grad_cam_mean = torch.mean(torch.tensor(grad_cam_result, device="cpu"), dim= (1,2)).cpu().numpy()
            print("grad cam mean shape: ", grad_cam_mean.shape)
            
            # Build visualizations in memory (no individual frame saves)
            visualizations = []
            for j in range(30):
              visualization = show_cam_on_image(cv2.resize(np.float32(new_frames[j].cpu())/255.0, (224,224)),
                                                grad_cam_result[j],
                                              use_rgb=True)
              visualizations.append(visualization)

            ############ COCO GradCAM Evaluation ############
            if gradcam_evaluator is not None and matched_video_id is not None:
                eval_result = gradcam_evaluator.evaluate_video(
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
                for group_name in ["small_only", "small_large", "small_large_visual_cues"]:
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


            
            # Get predictions
            predictions = classifier.module.get_types(logits)
            print(predictions[i,0].shape)
            prediction_text = test_dataset.keywords[predictions[i,0].item()]
            ground_truth_text = caption_text[0]
            print(f"Prediction: {prediction_text}")
            print(f"Ground Truth: {ground_truth_text}")
            
            # Create low-res visualization video with GradCAM + ROI bboxes
            save_lowres_visualization_video(
                video_directory=video_directory,
                video_name=video_timestamp,
                lowres_frames=new_frames,              # (30, H, W, C)
                gradcam_heatmaps=grad_cam_result,       # (30, 224, 224)
                gradcam_scores=grad_cam_mean,
                prediction_text=prediction_text,
                ground_truth_text=ground_truth_text,
                gradcam_evaluator=gradcam_evaluator,
                matched_video_id=matched_video_id if gradcam_evaluator else None,
                cam_threshold=args.cam_threshold,
            )
            
            i += 1
        all_predictions.append(predictions.cpu())
        del(frames)

    # Combine all predictions into a single tensor
    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions)

    ############## COCO Evaluation Summary ################
    if gradcam_evaluator is not None and all_eval_results:
        print("\n===== GradCAM Label-Group Evaluation Summary =====")
        # Aggregate requested cumulative label-group scores across videos.
        group_video_means = {
            "small_only": {"energy": [], "pointing": [], "iou": [], "frames": 0},
            "small_large": {"energy": [], "pointing": [], "iou": [], "frames": 0},
            "small_large_visual_cues": {"energy": [], "pointing": [], "iou": [], "frames": 0},
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
                    group_video_means[group_name]["energy"].append(gs['mean_energy_inside_bbox'])
                    has_group_data = True
                if not np.isnan(gs['mean_pointing_accuracy']):
                    group_video_means[group_name]["pointing"].append(gs['mean_pointing_accuracy'])
                if not np.isnan(gs['mean_iou']):
                    group_video_means[group_name]["iou"].append(gs['mean_iou'])
                group_video_means[group_name]["frames"] += int(gs.get('annotated_frames', 0))
            if has_group_data:
                videos_with_any_group += 1

        print(f"  Videos evaluated: {videos_with_any_group}")
        global_group_summary = {}
        for group_name, vals in group_video_means.items():
            if vals['energy']:
                global_group_summary[group_name] = {
                    "mean_energy_inside_bbox": float(np.mean(vals['energy'])),
                    "mean_pointing_accuracy": float(np.mean(vals['pointing'])),
                    "mean_iou": float(np.mean(vals['iou'])),
                    "annotated_frames": int(vals['frames']),
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
                    "mean_energy_inside_bbox": float("nan"),
                    "mean_pointing_accuracy": float("nan"),
                    "mean_iou": float("nan"),
                    "annotated_frames": 0,
                }
                print(f"  {group_name}: no matching annotated frames")

        # Optionally save detailed results
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
                        "per_video": {k: v for k, v in all_eval_results.items()},
                        "global_label_group_summary": global_group_summary,
                    },
                    _f, indent=2, default=_conv,
                )
            print(f"  Detailed results saved to {args.eval_output_json}")

    # # Save predictions to a file or process further
    # output_file = "predictions.pt"
    # torch.save(all_predictions, output_file)
    # print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
