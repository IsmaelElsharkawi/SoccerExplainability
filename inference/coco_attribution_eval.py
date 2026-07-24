import cv2
import json
from collections import defaultdict

import numpy as np


def _normalised_bbox_to_mask(bbox, height, width):
    """Convert a normalized [x, y, w, h] bbox to a binary mask (H, W)."""
    x, y, w, h = bbox
    x1, y1 = max(0, int(round(x * width))), max(0, int(round(y * height)))
    x2, y2 = min(width, int(round((x + w) * width))), min(height, int(round((y + h) * height)))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _combine_bbox_masks(bboxes, height, width):
    """Union of multiple normalized bbox masks."""
    combined = np.zeros((height, width), dtype=np.float32)
    for bbox in bboxes:
        combined = np.maximum(combined, _normalised_bbox_to_mask(bbox, height, width))
    return combined


def _spatial_auc(attribution_map, gt_mask):
    """[Step 2] Threshold-free spatial grounding score.

    Reviewers noted that S-IoU binarizes a continuous heatmap with an arbitrary
    cutoff, which can severely penalize models. This treats the attribution map
    as a continuous per-pixel prediction score and evaluates it against the
    discrete GT bbox mask via pixel-level ROC-AUC (no threshold). Returns NaN
    when the mask is all-foreground or all-background (AUC undefined).
    """
    from sklearn.metrics import roc_auc_score

    y_true = (gt_mask.flatten() > 0).astype(np.int32)
    positives = int(y_true.sum())
    if positives == 0 or positives == y_true.size:
        return float("nan")
    return float(roc_auc_score(y_true, attribution_map.flatten()))


def _compute_attribution_metrics_for_bboxes(attribution_map, bboxes, cam_threshold=0.5):
    """Compute Energy / Pointing / IoU / S-AUC for a given bbox subset."""
    height, width = attribution_map.shape[:2]
    gt_mask = _combine_bbox_masks(bboxes, height, width)

    total_energy = attribution_map.sum()
    if total_energy > 0:
        energy_inside_bbox = float((attribution_map * gt_mask).sum() / total_energy)
    else:
        energy_inside_bbox = 0.0

    peak_idx = np.unravel_index(np.argmax(attribution_map), attribution_map.shape)
    pointing_accuracy = float(gt_mask[peak_idx] > 0)

    cam_binary = (attribution_map >= cam_threshold * attribution_map.max()).astype(np.float32)
    intersection = (cam_binary * gt_mask).sum()
    union = np.clip(cam_binary + gt_mask, 0, 1).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    # [Step 2] Threshold-free complement to S-IoU.
    s_auc = _spatial_auc(attribution_map, gt_mask)

    return {
        "energy_inside_bbox": energy_inside_bbox,
        "pointing_accuracy": pointing_accuracy,
        "iou": iou,
        "s_auc": s_auc,
    }


def _frame_set_iou(gt_frames, pred_frames):
    """Compute IoU between two frame-index sets."""
    if not gt_frames and not pred_frames:
        return 1.0
    union = gt_frames | pred_frames
    if not union:
        return 0.0
    return float(len(gt_frames & pred_frames) / len(union))


# Threshold values used for the T-IoU sensitivity sweep (ratio of per-clip
# max frame score). Answers the reviewer request for a threshold sensitivity
# analysis instead of a single fixed adaptive cutoff.
TIOU_THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _temporal_tiou_sweep(frame_scores, gt_idx_set, thresholds=TIOU_THRESHOLD_SWEEP):
    """T-IoU as a function of the salient-frame threshold ratio.

    For each ratio r, predicted salient frames are those with mean intensity
    >= r * max(frame_scores); returns per-threshold T-IoU plus the mean over
    the sweep (a threshold-integrated summary of temporal overlap).
    """
    max_score = float(np.max(frame_scores)) if len(frame_scores) else 0.0
    per_threshold = {}
    values = []
    for ratio in thresholds:
        if max_score > 0:
            pred_idx_set = set(np.where(frame_scores >= ratio * max_score)[0].tolist())
        else:
            pred_idx_set = set()
        tiou = _frame_set_iou(gt_idx_set, pred_idx_set)
        per_threshold[f"{ratio:.1f}"] = float(tiou)
        values.append(float(tiou))
    return {
        "per_threshold": per_threshold,
        "mean_over_thresholds": float(np.mean(values)) if values else float("nan"),
    }


def _temporal_auc_ap(frame_scores, gt_idx_set, num_frames):
    """Threshold-free temporal localization scores.

    Treats the per-frame attribution intensity as a continuous saliency signal
    over time and scores it against the binary active-frame ground truth using
    ROC-AUC (temporal ranking quality) and Average Precision (top-of-ranking
    quality, robust to the sparse-positive regime). Returns NaN when only one
    class is present, since AUC/AP are undefined there.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.zeros(num_frames, dtype=np.int32)
    if gt_idx_set:
        y_true[list(gt_idx_set)] = 1
    positives = int(y_true.sum())
    if positives == 0 or positives == num_frames:
        return {"tAUC": float("nan"), "tAP": float("nan")}
    return {
        "tAUC": float(roc_auc_score(y_true, frame_scores)),
        "tAP": float(average_precision_score(y_true, frame_scores)),
    }


def _annotation_display_name(annotation, categories):
    return annotation.get("label") or categories.get(annotation.get("category_id"), "unknown")


def _annotation_type_key(annotation, categories):
    return categories.get(annotation.get("category_id"), annotation.get("label", "unknown"))


def _color_for_type(type_key):
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
    height, width = frame.shape[:2]
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1

    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        x1, y1 = int(round(x * width)), int(round(y * height))
        x2, y2 = int(round((x + w) * width)), int(round((y + h) * height))
        color = _color_for_type(_annotation_type_key(annotation, categories))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label_text = _annotation_display_name(annotation, categories)
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        text_x = max(0, min(x1, width - text_w - 6))

        if y1 - text_h - baseline - 8 >= 0:
            text_y = y1 - 6
            box_top = text_y - text_h - baseline - 2
            box_bottom = text_y + 2
        else:
            text_y = min(height - baseline - 2, y1 + text_h + baseline + 6)
            box_top = text_y - text_h - baseline - 2
            box_bottom = text_y + 2

        box_top = max(0, box_top)
        box_bottom = min(height - 1, box_bottom)
        box_right = min(width - 1, text_x + text_w + 6)

        cv2.rectangle(out, (text_x, box_top), (box_right, box_bottom), color, -1)
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
            cv2.putText(out, f"{key}: {val:.3f}", (position[0], y), font, 0.5, (255, 255, 255), 1)
            y += 20
    return out


class CocoAttributionEvaluator:
    """Load COCO annotations and evaluate attribution heatmaps against them."""

    def __init__(self, coco_json_path):
        with open(coco_json_path, "r") as file_obj:
            self.coco = json.load(file_obj)

        self.images_by_id = {img["id"]: img.copy() for img in self.coco["images"]}
        self.image_id_by_video_second = {
            (img["video_id"], img["second"]): img["id"] for img in self.coco["images"]
        }

        video_dims = {}
        for img in self.coco["images"]:
            video_id = img["video_id"]
            if video_id not in video_dims and img["width"] > 1 and img["height"] > 1:
                video_dims[video_id] = (img["width"], img["height"])

        for img in self.images_by_id.values():
            if img["width"] <= 1 or img["height"] <= 1:
                video_id = img["video_id"]
                if video_id in video_dims:
                    img["width"], img["height"] = video_dims[video_id]

        self.annots_by_image = defaultdict(list)
        for ann in self.coco["annotations"]:
            img = self.images_by_id.get(ann["image_id"])
            if not img:
                continue

            ann_copy = ann.copy()
            if not ann.get("no_roi", False) and img["width"] > 1 and img["height"] > 1:
                img_w, img_h = img["width"], img["height"]
                bx, by, bw, bh = ann["bbox"]
                if not (bx <= 1.0 and by <= 1.0 and (bx + bw) <= 1.0 and (by + bh) <= 1.0):
                    ann_copy["bbox"] = [bx / img_w, by / img_h, bw / img_w, bh / img_h]
            self.annots_by_image[ann["image_id"]].append(ann_copy)

        self.categories = {cat["id"]: cat["name"] for cat in self.coco.get("categories", [])}

    def get_annotations_for_frame(self, video_id, second):
        image_id = self.image_id_by_video_second.get((video_id, second))
        if image_id is None:
            return None
        annots = self.annots_by_image.get(image_id)
        return annots if annots else None

    def has_annotation(self, video_id, second):
        return (video_id, second) in self.image_id_by_video_second

    def evaluate_frame(self, attribution_map, video_id, second, iou_threshold=0.5, cam_threshold=0.5):
        annots = self.get_annotations_for_frame(video_id, second)
        if annots is None:
            return None

        height, width = attribution_map.shape[:2]
        roi_annots = [a for a in annots if not a.get("no_roi", False)]
        is_no_roi = len(roi_annots) == 0
        metrics = {"has_roi": 0.0 if is_no_roi else 1.0}

        if is_no_roi:
            flat = attribution_map.flatten()
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
            metrics["s_auc"] = float("nan")  # [Step 2]
            return metrics

        bboxes = [a["bbox"] for a in roi_annots]
        metrics.update(_compute_attribution_metrics_for_bboxes(attribution_map, bboxes, cam_threshold=cam_threshold))

        per_ann = []
        total_energy = attribution_map.sum()
        cam_binary = (attribution_map >= cam_threshold * attribution_map.max()).astype(np.float32)
        peak_idx = np.unravel_index(np.argmax(attribution_map), attribution_map.shape)

        for ann in roi_annots:
            ann_mask = _normalised_bbox_to_mask(ann["bbox"], height, width)
            ann_energy = (attribution_map * ann_mask).sum()
            ann_pointing = float(ann_mask[peak_idx] > 0)
            ann_inter = (cam_binary * ann_mask).sum()
            ann_union = np.clip(cam_binary + ann_mask, 0, 1).sum()
            per_ann.append(
                {
                    "annotation_id": ann["id"],
                    "category": self.categories.get(ann.get("category_id"), "unknown"),
                    "label": ann.get("label", ""),
                    "energy": float(ann_energy / total_energy) if total_energy > 0 else 0.0,
                    "pointing": ann_pointing,
                    "iou": float(ann_inter / ann_union) if ann_union > 0 else 0.0,
                }
            )
        metrics["per_annotation"] = per_ann

        def _norm_label(annotation):
            return str(annotation.get("label", "")).strip().lower()

        group_to_labels = {
            "small_only": {"small label"},
            "small_large": {"small label", "large label"},
            "small_large_visual_cues": {"small label", "large label", "visual cue"},
        }

        group_scores = {}
        for group_name, allowed_labels in group_to_labels.items():
            group_bboxes = [a["bbox"] for a in roi_annots if _norm_label(a) in allowed_labels]
            if group_bboxes:
                group_scores[group_name] = _compute_attribution_metrics_for_bboxes(
                    attribution_map, group_bboxes, cam_threshold=cam_threshold
                )
                group_scores[group_name]["frames_with_group"] = 1
            else:
                group_scores[group_name] = {
                    "energy_inside_bbox": float("nan"),
                    "pointing_accuracy": float("nan"),
                    "iou": float("nan"),
                    "s_auc": float("nan"),  # [Step 2]
                    "frames_with_group": 0,
                }
        metrics["label_group_scores"] = group_scores
        return metrics

    def evaluate_video(self, attribution_maps, video_id, start_second=0, iou_threshold=0.5, cam_threshold=0.5):
        num_frames = attribution_maps.shape[0]
        per_frame = []
        energies, pointings, ious = [], [], []
        s_aucs = []  # [Step 2]
        group_acc = {
            "small_only": {"energy": [], "pointing": [], "iou": [], "s_auc": []},
            "small_large": {"energy": [], "pointing": [], "iou": [], "s_auc": []},
            "small_large_visual_cues": {"energy": [], "pointing": [], "iou": [], "s_auc": []},
        }

        for frame_idx in range(num_frames):
            second = start_second + frame_idx
            frame_metrics = self.evaluate_frame(
                attribution_maps[frame_idx],
                video_id,
                second,
                iou_threshold=iou_threshold,
                cam_threshold=cam_threshold,
            )
            per_frame.append({"second": second, "metrics": frame_metrics})

            if frame_metrics is not None and frame_metrics["has_roi"]:
                energies.append(frame_metrics["energy_inside_bbox"])
                pointings.append(frame_metrics["pointing_accuracy"])
                ious.append(frame_metrics["iou"])
                if not np.isnan(frame_metrics.get("s_auc", np.nan)):  # [Step 2]
                    s_aucs.append(frame_metrics["s_auc"])

                for group_name, group_metrics in frame_metrics.get("label_group_scores", {}).items():
                    if not np.isnan(group_metrics.get("energy_inside_bbox", np.nan)):
                        group_acc[group_name]["energy"].append(group_metrics["energy_inside_bbox"])
                    if not np.isnan(group_metrics.get("pointing_accuracy", np.nan)):
                        group_acc[group_name]["pointing"].append(group_metrics["pointing_accuracy"])
                    if not np.isnan(group_metrics.get("iou", np.nan)):
                        group_acc[group_name]["iou"].append(group_metrics["iou"])
                    if not np.isnan(group_metrics.get("s_auc", np.nan)):  # [Step 2]
                        group_acc[group_name]["s_auc"].append(group_metrics["s_auc"])

        if energies:
            summary = {
                "mean_energy_inside_bbox": float(np.mean(energies)),
                "mean_pointing_accuracy": float(np.mean(pointings)),
                "mean_iou": float(np.mean(ious)),
                "mean_s_auc": float(np.mean(s_aucs)) if s_aucs else float("nan"),  # [Step 2]
            }
        else:
            summary = {
                "mean_energy_inside_bbox": float("nan"),
                "mean_pointing_accuracy": float("nan"),
                "mean_iou": float("nan"),
                "mean_s_auc": float("nan"),  # [Step 2]
            }

        summary["annotated_frames"] = len(energies)
        summary["total_frames"] = num_frames
        summary["label_group_scores"] = {}

        for group_name, vals in group_acc.items():
            if vals["energy"]:
                summary["label_group_scores"][group_name] = {
                    "mean_energy_inside_bbox": float(np.mean(vals["energy"])),
                    "mean_pointing_accuracy": float(np.mean(vals["pointing"])),
                    "mean_iou": float(np.mean(vals["iou"])),
                    "mean_s_auc": float(np.mean(vals["s_auc"])) if vals["s_auc"] else float("nan"),  # [Step 2]
                    "annotated_frames": len(vals["energy"]),
                }
            else:
                summary["label_group_scores"][group_name] = {
                    "mean_energy_inside_bbox": float("nan"),
                    "mean_pointing_accuracy": float("nan"),
                    "mean_iou": float("nan"),
                    "mean_s_auc": float("nan"),  # [Step 2]
                    "annotated_frames": 0,
                }

        # Temporal localization from frame-level attribution intensity.
        # Uses cumulative label groups as temporal tiers.
        flat_maps = attribution_maps.reshape(num_frames, -1)
        frame_scores = np.mean(flat_maps, axis=1)
        max_score = float(np.max(frame_scores)) if len(frame_scores) else 0.0
        score_threshold = cam_threshold * max_score
        if max_score > 0:
            pred_idx_set = set(np.where(frame_scores >= score_threshold)[0].tolist())
        else:
            pred_idx_set = set()

        gt_idx_by_tier = {
            "small_only": set(),
            "small_large": set(),
            "small_large_visual_cues": set(),
        }

        for idx, frame_item in enumerate(per_frame):
            frame_metrics = frame_item["metrics"]
            if frame_metrics is None:
                continue
            for tier_name in gt_idx_by_tier:
                tier_metrics = frame_metrics.get("label_group_scores", {}).get(tier_name, {})
                if int(tier_metrics.get("frames_with_group", 0)) > 0:
                    gt_idx_by_tier[tier_name].add(idx)

        temporal_tier_scores = {}
        valid_tious = []
        valid_taucs = []
        valid_taps = []
        valid_sweep_means = []

        for tier_name in ["small_only", "small_large", "small_large_visual_cues"]:
            gt_idx_set = gt_idx_by_tier[tier_name]

            if len(gt_idx_set) == 0:
                temporal_tier_scores[tier_name] = {
                    "tIoU": float("nan"),
                    "tAUC": float("nan"),
                    "tAP": float("nan"),
                    "tIoU_sweep": None,
                    "gt_frames": 0,
                    "pred_frames": int(len(pred_idx_set)),
                }
                continue

            tier_tiou = _frame_set_iou(gt_idx_set, pred_idx_set)
            tier_sweep = _temporal_tiou_sweep(frame_scores, gt_idx_set)
            tier_auc_ap = _temporal_auc_ap(frame_scores, gt_idx_set, num_frames)

            temporal_tier_scores[tier_name] = {
                "tIoU": float(tier_tiou),
                "tAUC": tier_auc_ap["tAUC"],
                "tAP": tier_auc_ap["tAP"],
                "tIoU_sweep": tier_sweep,
                "gt_frames": int(len(gt_idx_set)),
                "pred_frames": int(len(pred_idx_set)),
            }
            valid_tious.append(float(tier_tiou))
            if not np.isnan(tier_auc_ap["tAUC"]):
                valid_taucs.append(tier_auc_ap["tAUC"])
            if not np.isnan(tier_auc_ap["tAP"]):
                valid_taps.append(tier_auc_ap["tAP"])
            if not np.isnan(tier_sweep["mean_over_thresholds"]):
                valid_sweep_means.append(tier_sweep["mean_over_thresholds"])

        summary["temporal_localization"] = {
            "score_threshold_ratio": float(cam_threshold),
            "score_threshold": float(score_threshold),
            "tiers": temporal_tier_scores,
            "mean_tIoU": float(np.mean(valid_tious)) if valid_tious else float("nan"),
            "mean_tIoU_sweep": float(np.mean(valid_sweep_means)) if valid_sweep_means else float("nan"),
            "mean_tAUC": float(np.mean(valid_taucs)) if valid_taucs else float("nan"),
            "mean_tAP": float(np.mean(valid_taps)) if valid_taps else float("nan"),
        }

        return {"per_frame": per_frame, "summary": summary}

    def get_annotated_video_ids(self):
        return sorted(set(img["video_id"] for img in self.coco["images"]))


# Backward-compatible class alias for existing imports.
CocoGradCamEvaluator = CocoAttributionEvaluator

__all__ = [
    "CocoAttributionEvaluator",
    "CocoGradCamEvaluator",
    "draw_gt_bboxes_on_frame",
    "overlay_metrics_text",
]
