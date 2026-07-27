"""[Step 4b] S-IoU sensitivity curve, recomputed from raw saliency maps.

S-AUC (see ``coco_attribution_eval._spatial_auc``) is the integral of S-IoU
over the same 0.50-0.95 threshold sweep T-IoU uses, but only the integrated
scalar is persisted to the ``<clip>.json`` sidecars written by
``save_clip_saliency_and_scores`` -- the per-threshold curve behind it is
discarded after use. Unlike the temporal side, where ``tIoU_sweep.per_threshold``
is saved directly, there is no sidecar field to read this off of.

This script recomputes it from source instead: the raw per-frame saliency
maps saved in each clip's ``.npz`` sidecar (``--load_npz`` in
convergence_analysis.py only reads the 1-D ``frame_scores`` array from the
same file; this reads the 2-D ``saliency_maps`` array), re-evaluated against
the ROI boxes in the COCO annotation file at each ratio in the sweep, using
the project's own ``_spatial_iou_at``/``_combine_bbox_masks`` so the result
is methodologically identical to how S-AUC itself is computed, just without
collapsing the curve into one number.

Example:
    python inference/spatial_sensitivity_sweep.py \\
        --saliency_dir <run>/saliency \\
        --annotations_coco_json annotations-coco.json \\
        --output_csv <run>/saliency/analysis/spatial_sensitivity.csv
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from coco_attribution_eval import (
    CocoAttributionEvaluator, _combine_bbox_masks, _spatial_iou_at, THRESHOLD_SWEEP,
)

TIER_TO_LABELS = {
    "small_only": {"small label"},
    "small_large": {"small label", "large label"},
    "small_large_visual_cues": {"small label", "large label", "visual cue"},
}


def _norm_label(annotation):
    return str(annotation.get("label", "")).strip().lower()


def clip_siou_sweep(evaluator, clip_id, saliency_maps, allowed_labels):
    """Per-clip mean S-IoU at each threshold ratio (mirrors mean_iou's own
    frame-averaging, just parameterized by ratio instead of a fixed 0.5)."""
    per_ratio_frame_vals = {f"{r:.2f}": [] for r in THRESHOLD_SWEEP}
    for frame_idx in range(saliency_maps.shape[0]):
        annots = evaluator.get_annotations_for_frame(clip_id, frame_idx)
        if not annots:
            continue
        roi_annots = [a for a in annots if not a.get("no_roi", False)]
        group_bboxes = [a["bbox"] for a in roi_annots if _norm_label(a) in allowed_labels]
        if not group_bboxes:
            continue
        amap = saliency_maps[frame_idx]
        gt_mask = _combine_bbox_masks(group_bboxes, *amap.shape[:2])
        for r in THRESHOLD_SWEEP:
            per_ratio_frame_vals[f"{r:.2f}"].append(_spatial_iou_at(amap, gt_mask, r))
    if not per_ratio_frame_vals[f"{THRESHOLD_SWEEP[0]:.2f}"]:
        return None
    return {th: float(np.mean(vals)) for th, vals in per_ratio_frame_vals.items()}


def spatial_sensitivity_sweep(saliency_dir, annotations_coco_json, tier):
    evaluator = CocoAttributionEvaluator(annotations_coco_json)
    allowed_labels = TIER_TO_LABELS[tier]

    per_thresh_clip_means = {f"{r:.2f}": [] for r in THRESHOLD_SWEEP}
    n_clips_total, n_clips_used = 0, 0
    for jf in sorted(glob.glob(os.path.join(saliency_dir, "*.json"))):
        n_clips_total += 1
        meta = json.load(open(jf))
        clip_id = meta.get("clip_id", os.path.splitext(os.path.basename(jf))[0])
        npz_path = os.path.splitext(jf)[0] + ".npz"
        if not os.path.isfile(npz_path):
            continue
        with np.load(npz_path) as d:
            if "saliency_maps" not in d.files:
                continue
            saliency_maps = d["saliency_maps"]
        clip_result = clip_siou_sweep(evaluator, clip_id, saliency_maps, allowed_labels)
        if clip_result is None:
            continue
        n_clips_used += 1
        for th, v in clip_result.items():
            per_thresh_clip_means[th].append(v)

    per_threshold = {th: float(np.mean(vals)) if vals else float("nan")
                     for th, vals in per_thresh_clip_means.items()}
    return per_threshold, n_clips_used, n_clips_total


def write_csv(per_threshold, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_s_iou_pct"])
        for th in [f"{r:.2f}" for r in THRESHOLD_SWEEP]:
            writer.writerow([th, 100 * per_threshold[th]])


def main():
    parser = argparse.ArgumentParser(
        description="[Step 4b] S-IoU vs. salient-region threshold, recomputed from raw saliency maps.")
    parser.add_argument("--saliency_dir", type=str, required=True,
                        help="Directory of per-clip artifacts saved during inference (Step 3), "
                             "including the .npz sidecars with 'saliency_maps'.")
    parser.add_argument("--annotations_coco_json", type=str, default="annotations-coco.json",
                        help="COCO-format ROI annotations to score against.")
    parser.add_argument("--tier", type=str, default="small_large_visual_cues",
                        choices=list(TIER_TO_LABELS),
                        help="Cue tier whose ROI labels are included (default: P+S+C).")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Where to write the per-threshold CSV (default: <saliency_dir>/analysis/"
                             "spatial_sensitivity.csv).")
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(args.saliency_dir, "analysis", "spatial_sensitivity.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    per_threshold, n_used, n_total = spatial_sensitivity_sweep(
        args.saliency_dir, args.annotations_coco_json, args.tier)

    print(f"[Step 4b] {n_used}/{n_total} clips contributed at least one annotated frame "
          f"(tier={args.tier}).")
    print("\n[Step 4b] S-IoU sensitivity (mean %, tier={}):".format(args.tier))
    for r in THRESHOLD_SWEEP:
        th = f"{r:.2f}"
        print(f"  {th}: {100 * per_threshold[th]:.3f}")

    write_csv(per_threshold, output_csv)
    print(f"\n[Step 4b] Output written to {output_csv}")


if __name__ == "__main__":
    main()
