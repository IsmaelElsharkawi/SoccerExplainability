"""[Step 4c] AUC-ROC / Average Precision, recomputed post-inference from
saved saliency artifacts -- spatial AND temporal -- using scikit-learn
directly.

This is a standalone, read-only script. It does not import or call any
model. It only reads what inference already wrote to Drive: each clip's
`<clip>.npz` (raw per-frame saliency maps) and `<clip>.json` (clip_id,
ground truth), plus `annotations-coco.json`. Point it at a finished run and
it recomputes what reviewer fUPK actually asked for (W3/Q2): *"a
threshold-free metric, such as the Area Under the ROC Curve (AUC), to
evaluate the continuous prediction scores... against the discrete
ground-truth bounding boxes,"* via `sklearn.metrics.roc_auc_score` --
plus `average_precision_score` (AP) alongside it.

Why recompute at all instead of reading `mean_s_auc` / `mean_tAUC` out of the
existing run artifacts: both of those are threshold-sweep IoU integrals, not
rank-based scores (see the STALE warning at the top of REBUTTAL_TABLES.md --
"previously all three called sklearn.roc_auc_score / average_precision_score
... they are now integrals of the IoU-vs-threshold curve"). On that
definition 0.5 is not chance, so neither ever actually answered fUPK's
request; they're the same threshold-dependent quantity as S-IoU/T-IoU, just
averaged over ten cutoffs instead of one. This script goes back to calling
sklearn directly, spatially and temporally:

  * **Spatial** -- one call per annotated frame: every pixel's attribution
    value is the score, "inside a tier-relevant cue box" is the label.
  * **Temporal** -- one call per clip: each of the clip's frames is a
    sample, the per-frame mean heatmap intensity is the score, "frame
    carries a tier-relevant cue box" is the label. There is no equivalent of
    this anywhere in the existing pipeline -- `T-AUC`/`T-AP` in
    `coco_attribution_eval.py` are still the threshold-sweep version.

Chance AUROC is 0.5 in both cases; chance AP is the positive prevalence (box
pixel fraction, or annotated-frame fraction).

Paste into a Colab cell after mounting Drive (scikit-learn is preinstalled
on stock Colab runtimes; if not, `!pip install -q scikit-learn` first):

    from google.colab import drive
    drive.mount('/content/drive')

    # ... paste this file below, then in the next cell:
    results = run()
    results[('chefer_siglip', 'run_0029')]['summary']['small_large_visual_cues']

From a shell, once Drive is mounted:

    pip install scikit-learn  # if not already available
    python inference/spatial_auc_recompute.py

Needs `annotations-coco.json` reachable in the runtime -- it is NOT expected
to already be on Drive. ANNOTATIONS_JSON below defaults to the path
`gradcam_inference.py` uses when the repo is checked out in Colab
(`/content/SoccerExplainability/annotations-coco.json`); if you don't have
the repo cloned there, upload just that one file (3 MB) anywhere reachable
and point ANNOTATIONS_JSON at it before running.

The annotation-loading (`CocoLite`) is still duplicated from
`coco_attribution_eval.py` rather than imported -- that's just COCO-JSON
indexing, not the metric under review -- so this still survives being pasted
into a bare Colab cell with no repository checked out. Depends on numpy and
scikit-learn.
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Hard-coded run configuration.
# ---------------------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/SoccerExplainability-output"
ANNOTATIONS_JSON = "/content/SoccerExplainability/annotations-coco.json"

# Per-clip rows for every run, and one small consolidated table. Set either
# to None to skip it; the per-run CSVs under Drive are written regardless.
COMBINED_CLIPS_CSV = "/content/grounding_auc_clips.csv"
COMBINED_SUMMARY_CSV = "/content/grounding_auc_summary.csv"

TIER_TO_LABELS = {
    "small_only": {"small label"},
    "small_large": {"small label", "large label"},
    "small_large_visual_cues": {"small label", "large label", "visual cue"},
}
TIER_ORDER = ["small_only", "small_large", "small_large_visual_cues"]
TIER_SHORT = {"small_only": "P", "small_large": "P+S", "small_large_visual_cues": "P+S+C"}


# ---------------------------------------------------------------------------
# Duplicated from coco_attribution_eval.py: annotation indexing.
# ---------------------------------------------------------------------------
class CocoLite:
    """Minimal reimplementation of CocoAttributionEvaluator's annotation
    lookup (video_id, second) -> boxes, including the same bbox
    normalization and per-video width/height backfill for degenerate
    (<=1px) image records."""

    def __init__(self, coco_json_path):
        with open(coco_json_path) as f:
            coco = json.load(f)

        self.images_by_id = {img["id"]: img.copy() for img in coco["images"]}
        self.image_id_by_video_second = {
            (img["video_id"], img["second"]): img["id"] for img in coco["images"]
        }

        video_dims = {}
        for img in coco["images"]:
            vid = img["video_id"]
            if vid not in video_dims and img["width"] > 1 and img["height"] > 1:
                video_dims[vid] = (img["width"], img["height"])
        for img in self.images_by_id.values():
            if img["width"] <= 1 or img["height"] <= 1:
                vid = img["video_id"]
                if vid in video_dims:
                    img["width"], img["height"] = video_dims[vid]

        self.annots_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            img = self.images_by_id.get(ann["image_id"])
            if not img:
                continue
            ann = ann.copy()
            if not ann.get("no_roi", False) and img["width"] > 1 and img["height"] > 1:
                img_w, img_h = img["width"], img["height"]
                bx, by, bw, bh = ann["bbox"]
                if not (bx <= 1.0 and by <= 1.0 and (bx + bw) <= 1.0 and (by + bh) <= 1.0):
                    ann["bbox"] = [bx / img_w, by / img_h, bw / img_w, bh / img_h]
            self.annots_by_image[ann["image_id"]].append(ann)

    def annotations_for(self, video_id, second):
        image_id = self.image_id_by_video_second.get((video_id, second))
        return self.annots_by_image.get(image_id) if image_id is not None else None


def _normalised_bbox_to_mask(bbox, height, width):
    x, y, w, h = bbox
    x1, y1 = max(0, int(round(x * width))), max(0, int(round(y * height)))
    x2, y2 = min(width, int(round((x + w) * width))), min(height, int(round((y + h) * height)))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _combine_bbox_masks(bboxes, height, width):
    combined = np.zeros((height, width), dtype=np.float32)
    for bbox in bboxes:
        combined = np.maximum(combined, _normalised_bbox_to_mask(bbox, height, width))
    return combined


def _norm_label(ann):
    return str(ann.get("label", "")).strip().lower()


# ---------------------------------------------------------------------------
# Both metrics are direct scikit-learn calls. sklearn requires both classes
# present, so each wrapper checks that itself first and returns (NaN, NaN)
# rather than letting sklearn raise -- happens routinely when a tier has no
# annotated frame in a clip, or (temporally) when every frame is annotated.
# ---------------------------------------------------------------------------
def _spatial_pixel_auc(attribution_map, gt_mask):
    """Pixel-level AUROC/AP for one frame's heatmap vs. its cue-box mask:
    every pixel's attribution value is the score passed to
    `sklearn.metrics.roc_auc_score` / `average_precision_score`, "inside a
    tier-relevant cue box" is the label. This is exactly the metric fUPK
    described (W3/Q2)."""
    labels = (gt_mask > 0).astype(np.int8).ravel()
    scores = np.asarray(attribution_map, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _temporal_frame_auc(frame_scores, gt_idx_set, num_frames):
    """Frame-level AUROC/AP for one clip's temporal salience vs. its
    annotated-frame set -- the same sklearn calls, applied to the clip's 30
    per-frame mean heatmap intensities instead of pixels. Label = 1 if the
    frame carries a cue box in this tier (regardless of whether that frame
    has any COCO annotation record at all -- unannotated frames are
    negatives, matching how the ground-truth set is built for the pipeline's
    own T-IoU/T-AUC)."""
    y_true = np.zeros(num_frames, dtype=np.int8)
    if gt_idx_set:
        y_true[list(gt_idx_set)] = 1
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == num_frames:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, frame_scores)), float(average_precision_score(y_true, frame_scores))


# ---------------------------------------------------------------------------
# Per-clip / per-run recompute.
# ---------------------------------------------------------------------------
def clip_gt_idx_by_tier(coco, clip_id, num_frames):
    """Which frame indices carry a tier-relevant cue box, per tier --
    mirrors coco_attribution_eval's gt_idx_by_tier construction exactly."""
    gt_idx_by_tier = {tier: set() for tier in TIER_ORDER}
    for frame_idx in range(num_frames):
        annots = coco.annotations_for(clip_id, frame_idx)
        if not annots:
            continue
        roi = [a for a in annots if not a.get("no_roi", False)]
        labels_here = {_norm_label(a) for a in roi}
        for tier in TIER_ORDER:
            if labels_here & TIER_TO_LABELS[tier]:
                gt_idx_by_tier[tier].add(frame_idx)
    return gt_idx_by_tier


def clip_tier_spatial_auc(coco, clip_id, saliency_maps, allowed_labels):
    """Per-clip mean pixel AUROC/AUPRC for one tier, averaged over annotated
    frames that have at least one box in that tier (frame, then clip,
    never pooled -- matches the pipeline's own macro-averaging)."""
    aurocs, auprcs = [], []
    for frame_idx in range(saliency_maps.shape[0]):
        annots = coco.annotations_for(clip_id, frame_idx)
        if not annots:
            continue
        roi = [a for a in annots if not a.get("no_roi", False)]
        boxes = [a["bbox"] for a in roi if _norm_label(a) in allowed_labels]
        if not boxes:
            continue
        amap = saliency_maps[frame_idx]
        gt_mask = _combine_bbox_masks(boxes, *amap.shape[:2])
        auroc, auprc = _spatial_pixel_auc(amap, gt_mask)
        if not np.isnan(auroc):
            aurocs.append(auroc)
        if not np.isnan(auprc):
            auprcs.append(auprc)
    if not aurocs:
        return None
    return float(np.mean(aurocs)), float(np.mean(auprcs)), len(aurocs)


def analyse_run(coco, saliency_dir):
    """Recompute spatial + temporal AUROC/AP for every tier in one run.

    Returns per-(clip, tier) rows (for later bootstrapping/convergence work,
    same shape as the other Step 4 scripts) plus the run-level tier summary.
    """
    per_tier_spatial = {t: [] for t in TIER_ORDER}
    per_tier_temporal = {t: [] for t in TIER_ORDER}
    clip_rows = []
    n_clips = 0

    for jf in sorted(glob.glob(os.path.join(saliency_dir, "*.json"))):
        npz_path = os.path.splitext(jf)[0] + ".npz"
        if not os.path.isfile(npz_path):
            continue
        with open(jf) as f:
            meta = json.load(f)
        clip_id = meta.get("clip_id", os.path.splitext(os.path.basename(jf))[0])
        event_class = meta.get("ground_truth_text") or "unknown"
        with np.load(npz_path) as d:
            if "saliency_maps" not in d.files:
                continue
            saliency_maps = d["saliency_maps"]
        num_frames = saliency_maps.shape[0]
        frame_scores = saliency_maps.reshape(num_frames, -1).mean(axis=1)
        n_clips += 1

        gt_idx_by_tier = clip_gt_idx_by_tier(coco, clip_id, num_frames)

        for tier in TIER_ORDER:
            spatial = clip_tier_spatial_auc(coco, clip_id, saliency_maps, TIER_TO_LABELS[tier])
            t_auroc, t_ap = _temporal_frame_auc(frame_scores, gt_idx_by_tier[tier], num_frames)

            s_auroc, s_auprc, n_spatial_frames = spatial if spatial else (float("nan"), float("nan"), 0)
            if spatial:
                per_tier_spatial[tier].append((s_auroc, s_auprc))
            if not np.isnan(t_auroc):
                per_tier_temporal[tier].append((t_auroc, t_ap))

            clip_rows.append([
                clip_id, event_class, tier, n_spatial_frames, s_auroc, s_auprc,
                len(gt_idx_by_tier[tier]), num_frames, t_auroc, t_ap,
            ])

    summary = {}
    for tier in TIER_ORDER:
        svals = per_tier_spatial[tier]
        tvals = per_tier_temporal[tier]
        summary[tier] = {
            "mean_s_auroc": float(np.mean([v[0] for v in svals])) if svals else float("nan"),
            "mean_s_auprc": float(np.mean([v[1] for v in svals])) if svals else float("nan"),
            "n_clips_spatial": len(svals),
            "mean_t_auroc": float(np.mean([v[0] for v in tvals])) if tvals else float("nan"),
            "mean_t_ap": float(np.mean([v[1] for v in tvals])) if tvals else float("nan"),
            "n_clips_temporal": len(tvals),
        }

    return {"n_clips_total": n_clips, "summary": summary, "clip_rows": clip_rows}


CLIP_ROW_HEADER = ["clip_id", "event_class", "tier", "n_spatial_frames", "s_auroc", "s_auprc",
                   "n_gt_frames", "num_frames", "t_auroc", "t_ap"]
SUMMARY_HEADER = ["tier", "mean_s_auroc", "mean_s_auprc", "n_clips_spatial",
                  "mean_t_auroc", "mean_t_ap", "n_clips_temporal"]


def write_clip_rows_csv(clip_rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CLIP_ROW_HEADER)
        writer.writerows(clip_rows)


def write_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        for tier in TIER_ORDER:
            s = summary[tier]
            writer.writerow([tier, s["mean_s_auroc"], s["mean_s_auprc"], s["n_clips_spatial"],
                             s["mean_t_auroc"], s["mean_t_ap"], s["n_clips_temporal"]])


def print_summary(result):
    print(f"    clips with artifacts: {result['n_clips_total']}")
    header = (f"    {'tier':<10}{'S-AUROC':>10}{'S-AUPRC':>10}{'n':>5}"
              f"{'T-AUROC':>10}{'T-AP':>10}{'n':>5}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for tier in TIER_ORDER:
        s = result["summary"][tier]

        def fmt(v):
            return "--" if np.isnan(v) else f"{100 * v:.2f}"

        print(f"    {TIER_SHORT[tier]:<10}{fmt(s['mean_s_auroc']):>10}{fmt(s['mean_s_auprc']):>10}"
              f"{s['n_clips_spatial']:>5}{fmt(s['mean_t_auroc']):>10}{fmt(s['mean_t_ap']):>10}"
              f"{s['n_clips_temporal']:>5}")


def _short(experiment):
    """chefer_soccermaster_temporal -> soccermaster."""
    name = experiment
    if name.startswith("chefer_"):
        name = name[len("chefer_"):]
    if name.startswith("gradcam_"):
        name = name[len("gradcam_"):]
    for suffix in ("_temporal", "_spatial"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def find_runs(root):
    """Every <experiment>/run_XXXX/saliency directory under the output root."""
    runs = []
    for experiment in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, experiment)
        if not os.path.isdir(exp_dir):
            continue
        for run in sorted(os.listdir(exp_dir)):
            saliency_dir = os.path.join(exp_dir, run, "saliency")
            if os.path.isdir(saliency_dir):
                runs.append((experiment, run, saliency_dir))
    return runs


def write_combined_clips_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "run_id"] + CLIP_ROW_HEADER)
        for (experiment, run_id), result in sorted(results.items()):
            model = _short(experiment)
            for row in result["clip_rows"]:
                writer.writerow([experiment, model, run_id] + row)


def write_combined_summary_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "run_id"] + SUMMARY_HEADER)
        for (experiment, run_id), result in sorted(results.items()):
            model = _short(experiment)
            for tier in TIER_ORDER:
                s = result["summary"][tier]
                writer.writerow([experiment, model, run_id, tier, s["mean_s_auroc"],
                                 s["mean_s_auprc"], s["n_clips_spatial"], s["mean_t_auroc"],
                                 s["mean_t_ap"], s["n_clips_temporal"]])


def print_rebuttal_table(results):
    """One P/P+S/P+S+C row per model per metric, in the same layout already
    posted to fUPK -- so this can be dropped straight into a reply."""
    by_model = {}
    for (experiment, run_id), result in sorted(results.items()):
        by_model.setdefault(_short(experiment), result)

    models = sorted(by_model)
    header = "Metric (%) | " + " | ".join(
        f"{m} P | {m} P+S | {m} P+S+C" for m in models
    )
    print("\n[Step 4c] Rebuttal-ready table (rank-based AUROC/AP, spatial + temporal):\n")
    print(header)
    rows = (
        ("mean_s_auroc", "S-AUROC"),
        ("mean_s_auprc", "S-AUPRC"),
        ("mean_t_auroc", "T-AUROC"),
        ("mean_t_ap", "T-AP (rank)"),
    )
    for metric_key, metric_name in rows:
        row = [metric_name]
        for m in models:
            summary = by_model[m]["summary"]
            for tier in TIER_ORDER:
                v = summary[tier][metric_key]
                row.append("--" if np.isnan(v) else f"{100 * v:.2f}")
        print(" | ".join(row))
    print("\n(T-AP (rank) is the frame-ranking counterpart to the pipeline's own T-AP, "
          "which is still a threshold-sweep integral -- see module docstring.)")


def run(root=None, annotations_json=None, only=None, verbose=True,
        combined_clips_csv=COMBINED_CLIPS_CSV, combined_summary_csv=COMBINED_SUMMARY_CSV):
    """Recompute rank-based spatial + temporal AUROC/AP for every run under
    ``root``. Returns ``{(experiment, run): result}``. Nothing here re-runs a
    model -- only the raw saliency maps already on Drive and the COCO
    annotations are read.
    """
    root = root or DRIVE_ROOT
    annotations_json = annotations_json or ANNOTATIONS_JSON

    if not os.path.isdir(root):
        print(f"[Step 4c] Output root not found: {root}")
        print("          If this is Colab, mount Drive first:")
        print("              from google.colab import drive")
        print("              drive.mount('/content/drive')")
        return {}

    if not os.path.isfile(annotations_json):
        print(f"[Step 4c] annotations-coco.json not found at: {annotations_json}")
        print("          This file is not expected to already be on Drive -- upload it")
        print("          once (it's ~3 MB) and set ANNOTATIONS_JSON to wherever you put it.")
        print("          If you have the repo checked out in Colab, it's at")
        print("          <repo>/annotations-coco.json.")
        return {}

    coco = CocoLite(annotations_json)

    runs = find_runs(root)
    if only:
        runs = [r for r in runs if only in r[0] or only in r[1]]
    if not runs:
        print(f"[Step 4c] No <experiment>/run_XXXX/saliency directories under {root}"
              + (f" matching {only!r}." if only else "."))
        return {}

    print(f"[Step 4c] {len(runs)} run(s) under {root}.")

    results = {}
    for experiment, run_id, saliency_dir in runs:
        if verbose:
            print(f"\n[Step 4c] {experiment}/{run_id}")
        result = analyse_run(coco, saliency_dir)
        if result["n_clips_total"] == 0:
            print(f"    no clip artifacts in {saliency_dir}; skipped.")
            continue

        output_dir = os.path.join(saliency_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        write_summary_csv(result["summary"], os.path.join(output_dir, "grounding_auc_recomputed_summary.csv"))
        write_clip_rows_csv(result["clip_rows"], os.path.join(output_dir, "grounding_auc_recomputed_clips.csv"))

        if verbose:
            print_summary(result)
            print(f"    written to {output_dir}")

        results[(experiment, run_id)] = result

    if results and combined_clips_csv:
        parent = os.path.dirname(combined_clips_csv) or "."
        if os.path.isdir(parent):
            write_combined_clips_csv(results, combined_clips_csv)
            print(f"\n[Step 4c] Combined per-clip CSV -> {combined_clips_csv}")
    if results and combined_summary_csv:
        parent = os.path.dirname(combined_summary_csv) or "."
        if os.path.isdir(parent):
            write_combined_summary_csv(results, combined_summary_csv)
            print(f"[Step 4c] Combined summary CSV -> {combined_summary_csv}")

    if results:
        print_rebuttal_table(results)

    print(f"\n[Step 4c] Done: {len(results)} run(s).")
    return results


def main():
    return 0 if run() else 1


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_notebook():
        results = run()
    else:
        raise SystemExit(main())
