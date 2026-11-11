"""Table 1 grounding metrics, recomputed from saved per-clip saliency maps.

This is the single implementation behind every number in Table 1 of the
paper: Energy, Pointing, S-AUROC, S-AUPRC and T-IoU at the three cumulative
cue tiers (P, P+S, P+S+C). S-IoU (supplementary Table 5) comes for free.

It never touches a model. Its only inputs are the `<clip>.npz` /
`<clip>.json` pairs that every attribution script writes with
`--saliency_save_dir` (key `saliency_maps`, shape (T, h, w)), plus
`annotations-coco.json`. That makes every model, including runs produced
elsewhere, scored by exactly the same code.

Resolution
----------
Every model in the benchmark squashes the full broadcast frame into its
square input, so a token-grid map covers the whole frame. Two scoring modes
are supported:

* ``frame`` (default): each per-frame map is bilinearly upsampled to a common
  frame-shaped canvas (720x1280 by default) and the cue boxes are rasterized
  on that canvas. This matches the paper text (Sec. 4.2, Alg. 1) and removes
  the grid-size confound between models (14x14 ... 32x32 token grids).
* ``native``: maps are scored on their own token grid with the boxes
  rounded onto that grid. This is what the inference scripts' built-in
  evaluator has always done; it exists to validate against older numbers.

Aggregation
-----------
Spatial metrics are averaged over the frames of a clip that carry at least one
box of the tier, then over clips (clip-macro), which is how the inference
pipeline has always aggregated. T-IoU is computed per clip and averaged over
clips that have at least one annotated frame for the tier.
"""

import json
import os

import cv2
import numpy as np

from coco_attribution_eval import CocoAttributionEvaluator

TIERS = ("small_only", "small_large", "small_large_visual_cues")
TIER_SHORT = {"small_only": "P", "small_large": "P+S", "small_large_visual_cues": "P+S+C"}
TIER_LABELS = {
    "small_only": {"small label"},
    "small_large": {"small label", "large label"},
    "small_large_visual_cues": {"small label", "large label", "visual cue"},
}
SPATIAL_METRICS = ("energy", "pointing", "s_iou", "s_auroc", "s_auprc", "box_fraction")
TEMPORAL_METRICS = ("t_iou", "t_auroc", "t_auprc", "frame_fraction")

DEFAULT_CANVAS = (720, 1280)  # (height, width); SoccerNet broadcast frames are 16:9.
CAM_THRESHOLD = 0.5

# numpy 2 renamed trapz to trapezoid; Colab images ship either.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _norm_label(annotation):
    return str(annotation.get("label", "")).strip().lower()


def bbox_mask(bboxes, height, width):
    """Union mask of normalized [x, y, w, h] boxes on a (height, width) grid.

    Same rounding as `coco_attribution_eval._normalised_bbox_to_mask`, so the
    native mode reproduces the inference-time evaluator exactly.
    """
    mask = np.zeros((height, width), dtype=bool)
    for x, y, w, h in bboxes:
        x1, y1 = max(0, int(round(x * width))), max(0, int(round(y * height)))
        x2 = min(width, int(round((x + w) * width)))
        y2 = min(height, int(round((y + h) * height)))
        mask[y1:y2, x1:x2] = True
    return mask


def resize_map(attribution_map, height, width):
    """Bilinear upsampling of one map (half-pixel centers, like
    `F.interpolate(..., mode='bilinear', align_corners=False)`)."""
    if attribution_map.shape == (height, width):
        return attribution_map.astype(np.float32, copy=False)
    return cv2.resize(attribution_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def ranking_curves(scores_desc_ends, labels_sorted):
    """AUROC and AP from scores sorted in descending order.

    `scores_desc_ends` holds the indices of the last element of each run of
    tied scores. Collapsing ties to one operating point and integrating the
    ROC curve with trapezoids / the PR curve with the step rule matches
    `sklearn.metrics.roc_auc_score` and `average_precision_score`.
    """
    n_pos = int(labels_sorted.sum())
    n_neg = labels_sorted.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    tps = np.cumsum(labels_sorted, dtype=np.int64)[scores_desc_ends].astype(np.float64)
    fps = (scores_desc_ends + 1).astype(np.float64) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    auroc = float(_trapezoid(tpr, fpr))
    precision = tps / (tps + fps)
    recall = np.r_[0.0, tps / n_pos]
    auprc = float(np.sum(np.diff(recall) * precision))
    return auroc, auprc


def frame_tier_metrics(attribution_map, tier_masks, cam_threshold=CAM_THRESHOLD):
    """All spatial metrics of one frame for every tier present in `tier_masks`.

    The pixel ranking (the expensive part) is computed once per frame and
    shared by the tiers.
    """
    scores = attribution_map.ravel()
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    run_ends = np.r_[np.flatnonzero(np.diff(scores_sorted)), scores_sorted.size - 1]

    total = float(scores.sum())
    peak = int(np.argmax(scores))
    peak_val = float(scores[peak])
    salient = scores >= cam_threshold * peak_val if peak_val > 0 else np.zeros_like(scores, dtype=bool)

    out = {}
    for tier, mask in tier_masks.items():
        labels = mask.ravel()
        inside = float(scores[labels].sum())
        inter = np.count_nonzero(salient & labels)
        union = np.count_nonzero(salient | labels)
        auroc, auprc = ranking_curves(run_ends, labels[order])
        out[tier] = {
            "energy": inside / total if total > 0 else 0.0,
            "pointing": float(labels[peak]),
            "s_iou": inter / union if union else 0.0,
            "s_auroc": auroc,
            "s_auprc": auprc,
            "box_fraction": float(labels.mean()),
        }
    return out


def temporal_ranking(frame_scores, gt_frames):
    """T-AUROC and T-AUPRC: how well the per-frame mean attribution ranks the
    clip's annotated frames above its unannotated ones. Threshold-free, so
    unlike T-IoU it is not inflated by marking most frames salient; a flat
    map scores 0.5 AUROC and the annotated-frame fraction AUPRC."""
    scores = np.asarray(frame_scores, dtype=np.float64)
    labels = np.zeros(scores.size, dtype=bool)
    labels[list(gt_frames)] = True
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    run_ends = np.r_[np.flatnonzero(np.diff(scores_sorted)), scores_sorted.size - 1]
    return ranking_curves(run_ends, labels[order])


def temporal_iou(frame_scores, gt_frames, cam_threshold=CAM_THRESHOLD):
    """T-IoU between annotated frames and frames whose mean attribution is at
    least `cam_threshold` x the clip's maximum frame mean (paper Eq. 2)."""
    peak = float(np.max(frame_scores)) if len(frame_scores) else 0.0
    pred = set(np.flatnonzero(frame_scores >= cam_threshold * peak).tolist()) if peak > 0 else set()
    union = gt_frames | pred
    if not union:
        return float("nan"), len(pred)
    return len(gt_frames & pred) / len(union), len(pred)


def random_ranking_temporal_chance(gt_frames, num_frames, draws=2000, seed=0):
    """Expected T-IoU, T-AUROC and T-AUPRC of a clip whose frames get
    independent random scores. With only `num_frames` frames, the AP of a
    random ranking sits well above the annotated-frame fraction (a lucky early
    hit counts a lot), so the fraction is not a fair T-AUPRC baseline."""
    rng = np.random.default_rng(seed)
    tiou, auroc, auprc = [], [], []
    for _ in range(draws):
        scores = rng.random(num_frames)
        tiou.append(temporal_iou(scores, gt_frames)[0])
        roc, ap = temporal_ranking(scores, gt_frames)
        auroc.append(roc)
        auprc.append(ap)
    return float(np.mean(tiou)), float(np.mean(auroc)), float(np.mean(auprc))


class GroundingScorer:
    """Scores saved saliency maps against the SoccerLens cue boxes."""

    def __init__(self, coco_json, resolution="frame", canvas=DEFAULT_CANVAS, cam_threshold=CAM_THRESHOLD):
        if resolution not in ("frame", "native"):
            raise ValueError(f"resolution must be 'frame' or 'native', got {resolution!r}")
        self.coco = CocoAttributionEvaluator(coco_json)
        self.resolution = resolution
        self.canvas = tuple(canvas)
        self.cam_threshold = cam_threshold

    def annotated_clip_ids(self):
        return set(self.coco.get_annotated_video_ids())

    def _tier_boxes(self, clip_id, second):
        annots = self.coco.get_annotations_for_frame(clip_id, second) or []
        roi = [a for a in annots if not a.get("no_roi", False)]
        boxes = {}
        for tier in TIERS:
            tier_boxes = [a["bbox"] for a in roi if _norm_label(a) in TIER_LABELS[tier]]
            if tier_boxes:
                boxes[tier] = tier_boxes
        return boxes

    def gt_frame_sets(self, clip_id, num_frames):
        """{tier: set of frame indices carrying a box of that tier}."""
        sets = {tier: set() for tier in TIERS}
        for t in range(num_frames):
            for tier in self._tier_boxes(clip_id, t):
                sets[tier].add(t)
        return sets

    def score_clip(self, clip_id, saliency_maps):
        """Per-clip, per-tier metrics for one (T, h, w) stack of maps.

        Returns {tier: {metric: value, 'n_frames': int, 't_iou': float,
        'n_gt_frames': int, 'n_pred_frames': int}}; spatial values are NaN
        when the clip has no frame with a box of that tier.
        """
        maps = np.asarray(saliency_maps, dtype=np.float32)
        if maps.ndim != 3:
            raise ValueError(f"{clip_id}: expected saliency maps of shape (T, h, w), got {maps.shape}")
        num_frames = maps.shape[0]
        height, width = self.canvas if self.resolution == "frame" else maps.shape[1:]

        per_tier = {tier: {m: [] for m in SPATIAL_METRICS} for tier in TIERS}
        gt_frames = {tier: set() for tier in TIERS}
        frame_scores = np.zeros(num_frames, dtype=np.float64)

        for t in range(num_frames):
            frame_map = resize_map(maps[t], height, width)
            frame_scores[t] = float(frame_map.mean())
            boxes = self._tier_boxes(clip_id, t)
            if not boxes:
                continue
            masks = {tier: bbox_mask(b, height, width) for tier, b in boxes.items()}
            for tier, metrics in frame_tier_metrics(frame_map, masks, self.cam_threshold).items():
                gt_frames[tier].add(t)
                for name, value in metrics.items():
                    per_tier[tier][name].append(value)

        result = {}
        for tier in TIERS:
            row = {}
            for name, values in per_tier[tier].items():
                finite = [v for v in values if not np.isnan(v)]
                row[name] = float(np.mean(finite)) if finite else float("nan")
            row["n_frames"] = len(per_tier[tier]["energy"])
            if gt_frames[tier]:
                row["t_iou"], row["n_pred_frames"] = temporal_iou(frame_scores, gt_frames[tier], self.cam_threshold)
                row["t_auroc"], row["t_auprc"] = temporal_ranking(frame_scores, gt_frames[tier])
            else:
                row["t_iou"], row["n_pred_frames"] = float("nan"), 0
                row["t_auroc"], row["t_auprc"] = float("nan"), float("nan")
            # T-AUPRC chance: the clip's annotated-frame fraction.
            row["frame_fraction"] = len(gt_frames[tier]) / num_frames if gt_frames[tier] else float("nan")
            row["n_gt_frames"] = len(gt_frames[tier])
            result[tier] = row
        return result


def aggregate(clip_results):
    """Clip-macro mean of every metric per tier, ignoring NaN clips."""
    summary = {}
    for tier in TIERS:
        rows = [r[tier] for r in clip_results if tier in r]
        tier_summary = {}
        for name in SPATIAL_METRICS + TEMPORAL_METRICS:
            values = [r[name] for r in rows if not np.isnan(r[name])]
            tier_summary[name] = float(np.mean(values)) if values else float("nan")
            tier_summary[f"n_clips_{name}"] = len(values)
        tier_summary["n_frames"] = int(sum(r["n_frames"] for r in rows))
        summary[tier] = tier_summary
    return summary


def load_saliency_dir(saliency_dir, map_key="saliency_maps"):
    """Yield (clip_id, maps, meta) for every `<stem>.npz` with a `<stem>.json`
    sidecar in `saliency_dir`."""
    for name in sorted(os.listdir(saliency_dir)):
        if not name.endswith(".npz"):
            continue
        stem = name[:-4]
        meta_path = os.path.join(saliency_dir, stem + ".json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with np.load(os.path.join(saliency_dir, name)) as data:
            if map_key not in data.files:
                raise KeyError(f"{name} has no '{map_key}' array (available: {data.files})")
            maps = data[map_key]
        yield meta.get("clip_id", stem), maps, meta
