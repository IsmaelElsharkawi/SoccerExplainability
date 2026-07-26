"""[Step 4] Convergence analysis and per-class metric tables from saved artifacts.

This script belongs to instruction/step 4. It reads the per-clip saliency maps
and scores that were persisted during inference (instruction/step 3, via
``--saliency_save_dir``), then:

  1. Runs a *convergence analysis*: using bootstrap resampling over the set of
     evaluated clips, it estimates the mean of every grounding metric together
     with a 95% confidence interval, and traces how the estimate (and its CI
     half-width) stabilizes as more clips are included. This directly answers
     reviewer requests for confidence intervals and for evidence that the
     conclusions are stable despite the benchmark's small size.

  2. Builds a *per-class metrics table*: it groups clips by their soccer event
     class and reports mean +/- std (and clip counts) per metric, so that
     under-represented classes (e.g. Red Card) can be inspected explicitly.

Inputs are the ``<clip>.json`` metadata files written by
``save_clip_saliency_and_scores`` in ``inference_utils.py``. The ``<clip>.npz``
saliency maps are also loadable here (see ``--load_npz``) for future
frame-level analysis, but the metric tables are driven by the JSON summaries.

Example:
    python inference/convergence_analysis.py \\
        --saliency_dir output_chefer_matchvision_temporal/saliency \\
        --output_dir output_chefer_matchvision_temporal/convergence
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# [Step 4] Metric extraction: pull per-clip scalar metrics out of the saved
# summary JSON. Each entry maps a metric name to (kind, extractor).
# ---------------------------------------------------------------------------
TIERS = ["small_only", "small_large", "small_large_visual_cues"]


def _spatial_metrics(summary, tier):
    gs = (summary.get("label_group_scores", {}) or {}).get(tier, {}) or {}
    return {
        "Energy": gs.get("mean_energy_inside_bbox", np.nan),
        "Pointing": gs.get("mean_pointing_accuracy", np.nan),
        "S-IoU": gs.get("mean_iou", np.nan),
        # S-AUC is itself the threshold-integrated S-IoU, so there is no
        # separate "S-IoU (sweep)" column any more.
        "S-AUC": gs.get("mean_s_auc", np.nan),
    }


def _temporal_metrics(summary, tier):
    tl = summary.get("temporal_localization", {}) or {}
    ts = (tl.get("tiers", {}) or {}).get(tier, {}) or {}
    sweep = ts.get("tIoU_sweep") or {}
    return {
        "T-IoU": ts.get("tIoU", np.nan),
        "T-IoU (sweep)": sweep.get("mean_over_thresholds", np.nan),
        "T-AUC": ts.get("tAUC", np.nan),
        "T-AP": ts.get("tAP", np.nan),
    }


def extract_clip_metrics(summary, tier):
    """[Step 4] Combined spatial + temporal per-clip metric dict for a tier."""
    metrics = {}
    metrics.update(_spatial_metrics(summary, tier))
    metrics.update(_temporal_metrics(summary, tier))
    return metrics


METRIC_ORDER = ["Energy", "Pointing", "S-IoU", "S-AUC",
                "T-IoU", "T-IoU (sweep)", "T-AUC", "T-AP"]


# ---------------------------------------------------------------------------
# [Step 4] Loading saved artifacts and mapping clips to event classes.
# ---------------------------------------------------------------------------
def load_event_class_map(selected_videos_json):
    """[Step 4] Map video_id -> soccer event class from the selection file."""
    if not selected_videos_json or not os.path.isfile(selected_videos_json):
        return {}
    with open(selected_videos_json) as f:
        records = json.load(f)
    mapping = {}
    for rec in records:
        video = rec.get("video")
        caption = rec.get("caption")
        if video is not None and caption is not None:
            mapping[video] = str(caption)
    return mapping


def load_clip_records(saliency_dir, tier, event_map, load_npz=False):
    """[Step 4] Read every <clip>.json and assemble per-clip metric records."""
    records = []
    json_paths = sorted(glob.glob(os.path.join(saliency_dir, "*.json")))
    for path in json_paths:
        with open(path) as f:
            meta = json.load(f)
        clip_id = meta.get("clip_id", os.path.splitext(os.path.basename(path))[0])
        summary = meta.get("summary", {}) or {}
        event_class = event_map.get(clip_id) or meta.get("ground_truth_text") or "unknown"

        record = {
            "clip_id": clip_id,
            "event_class": event_class,
            "metrics": extract_clip_metrics(summary, tier),
        }

        if load_npz:
            npz_path = os.path.splitext(path)[0] + ".npz"
            if os.path.isfile(npz_path):
                with np.load(npz_path) as data:
                    record["frame_scores"] = data["frame_scores"]

        records.append(record)
    return records


# ---------------------------------------------------------------------------
# [Step 4] Convergence analysis via bootstrap resampling.
# ---------------------------------------------------------------------------
# Clip counts at which the bootstrap is evaluated. The analysis reports a CI
# half-width at each of these sample sizes only, rather than tracing every
# prefix n = 1..N.
BOOTSTRAP_SAMPLE_SIZES = [50, 100, 150, 200]


def _bootstrap_mean_ci(values, n_bootstrap, rng, alpha=0.05, sample_size=None):
    """Return (mean, ci_low, ci_high, ci_halfwidth) for a 1-D value array.

    ``sample_size`` is the size of each bootstrap resample. It defaults to the
    number of available clips, which is the ordinary bootstrap. Passing a
    smaller value answers "what CI would this metric have with that many
    clips?" — resamples are still drawn from *all* clips, so no subset of the
    data is privileged and there is no dependence on clip ordering.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if values.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    n = values.size
    m = n if sample_size is None else int(sample_size)
    if m <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    # Draw all resamples at once: (n_bootstrap, m) indices into `values`.
    boot_means = values[rng.integers(0, n, size=(n_bootstrap, m))].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (float(values.mean()), lo, hi, float((hi - lo) / 2.0))


def convergence_analysis(records, n_bootstrap, seed, tol,
                         sample_sizes=BOOTSTRAP_SAMPLE_SIZES, min_n=5):
    """[Step 4] Bootstrap CIs at full N plus a CI half-width per sample size.

    For each metric we compute the bootstrap mean/CI over all clips, then
    repeat the bootstrap at each requested sample size (default 50/100/150/200)
    and record the CI half-width. ``converged_at`` reports the smallest of
    those sizes at which the half-width drops to ``tol`` or below.

    A requested size larger than the number of clips carrying the metric is
    clamped to that count; the effective size travels with each curve point so
    the shortfall stays visible in the CSV and the printed table.
    """
    rng = np.random.default_rng(seed)

    final = {}
    curves = {}
    converged_at = {}

    for metric in METRIC_ORDER:
        all_vals = [rec["metrics"].get(metric, np.nan) for rec in records]
        n_avail = sum(1 for v in all_vals if not np.isnan(v))
        mean, lo, hi, half = _bootstrap_mean_ci(all_vals, n_bootstrap, rng)
        final[metric] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                         "ci_halfwidth": half, "n_clips": n_avail}

        curve = []
        conv_n = None
        for n in sorted(sample_sizes):
            # A metric can be NaN on a few clips (e.g. S-AUC is undefined when a
            # frame's cue mask covers everything), so the usable count can sit
            # just below a requested size. Clamp rather than drop the column,
            # and record the size actually used so the gap stays visible.
            n_used = min(int(n), n_avail)
            if n_used <= 0:
                continue
            if n_used == n_avail:
                # Same quantity as the full-sample bootstrap above; reuse it so
                # the two tables cannot disagree through resampling noise alone.
                sub_half = half
            else:
                _, _, _, sub_half = _bootstrap_mean_ci(
                    all_vals, n_bootstrap, rng, sample_size=n_used
                )
            curve.append((int(n), sub_half, n_used))
            if conv_n is None and n >= min_n and not np.isnan(sub_half) and sub_half <= tol:
                conv_n = int(n)
        curves[metric] = curve
        converged_at[metric] = conv_n

    return {"final": final, "curves": curves, "converged_at": converged_at}


# ---------------------------------------------------------------------------
# [Step 4] Per-class metrics table.
# ---------------------------------------------------------------------------
def per_class_table(records):
    """[Step 4] mean/std/count of each metric grouped by event class."""
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["event_class"], []).append(rec["metrics"])

    table = {}
    for event_class, metric_dicts in sorted(by_class.items()):
        row = {"count": len(metric_dicts)}
        for metric in METRIC_ORDER:
            vals = np.asarray(
                [m.get(metric, np.nan) for m in metric_dicts], dtype=np.float64
            )
            vals = vals[~np.isnan(vals)]
            if vals.size:
                row[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
            else:
                row[metric] = {"mean": float("nan"), "std": float("nan")}
        table[event_class] = row
    return table


# ---------------------------------------------------------------------------
# [Step 4] Output writers.
# ---------------------------------------------------------------------------
def _fmt(v, scale=100.0):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{scale * v:.2f}"


def write_per_class_csv(table, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["event_class", "count"]
        for metric in METRIC_ORDER:
            header += [f"{metric}_mean", f"{metric}_std"]
        writer.writerow(header)
        for event_class, row in table.items():
            line = [event_class, row["count"]]
            for metric in METRIC_ORDER:
                line += [row[metric]["mean"], row[metric]["std"]]
            writer.writerow(line)


def write_convergence_csv(result, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "ci_low", "ci_high", "ci_halfwidth",
                         "converged_at_n", "n_clips"])
        for metric in METRIC_ORDER:
            fin = result["final"][metric]
            writer.writerow([
                metric, fin["mean"], fin["ci_low"], fin["ci_high"],
                fin["ci_halfwidth"], result["converged_at"][metric], fin["n_clips"],
            ])


def write_convergence_curves_csv(result, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "n_clips", "ci_halfwidth", "n_clips_used"])
        for metric in METRIC_ORDER:
            for n, half, n_used in result["curves"][metric]:
                writer.writerow([metric, n, half, n_used])


def maybe_plot_convergence(result, path):
    """[Step 4] Optional convergence plot (skipped if matplotlib unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [Step 4] matplotlib not available; skipping convergence plot.")
        return
    plt.figure(figsize=(8, 5))
    for metric in METRIC_ORDER:
        curve = result["curves"][metric]
        if not curve:
            continue
        xs = [n for n, _, _ in curve]
        ys = [h for _, h, _ in curve]
        plt.plot(xs, ys, marker="o", label=metric)
    plt.xticks(sorted({n for m in METRIC_ORDER for n, _, _ in result["curves"][m]}))
    plt.xlabel("Number of clips (bootstrap resample size)")
    plt.ylabel("Bootstrap 95% CI half-width")
    plt.title("Metric convergence vs. number of clips")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_per_class_table(table):
    print("\n[Step 4] Per-class metrics (mean %, spatial+temporal):")
    header = f"{'event_class':<22}{'n':>4}  " + "  ".join(f"{m:>13}" for m in METRIC_ORDER)
    print(header)
    print("-" * len(header))
    for event_class, row in table.items():
        cells = "  ".join(f"{_fmt(row[m]['mean']):>13}" for m in METRIC_ORDER)
        print(f"{event_class:<22}{row['count']:>4}  {cells}")


def print_sample_size_table(result):
    """[Step 4] CI half-width at each requested sample size."""
    sizes = sorted({n for m in METRIC_ORDER for n, _, _ in result["curves"][m]})
    if not sizes:
        return
    print("\n[Step 4] Bootstrap 95% CI half-width by sample size (%):")
    header = f"{'metric':<16}" + "".join(f"{'n=' + str(n):>10}" for n in sizes)
    print(header)
    print("-" * len(header))
    notes = []
    for metric in METRIC_ORDER:
        got = {n: (h, u) for n, h, u in result["curves"][metric]}
        cells = ""
        for n in sizes:
            if n not in got:
                cells += f"{'--':>10}"
                continue
            half, used = got[n]
            mark = "*" if used != n else ""
            cells += f"{_fmt(half) + mark:>10}"
            if mark:
                notes.append(f"{metric} n={n} computed on {used} clips")
        print(f"{metric:<16}{cells}")
    for note in dict.fromkeys(notes):
        print(f"  * {note} (metric undefined on the remainder)")


def print_convergence(result):
    print("\n[Step 4] Convergence summary (bootstrap 95% CI over all clips):")
    header = f"{'metric':<16}{'mean':>10}{'ci_low':>10}{'ci_high':>10}{'half':>10}{'conv@n':>8}"
    print(header)
    print("-" * len(header))
    for metric in METRIC_ORDER:
        fin = result["final"][metric]
        conv = result["converged_at"][metric]
        print(f"{metric:<16}"
              f"{_fmt(fin['mean']):>10}{_fmt(fin['ci_low']):>10}{_fmt(fin['ci_high']):>10}"
              f"{_fmt(fin['ci_halfwidth']):>10}{('--' if conv is None else conv):>8}")


def main():
    # [Step 4] CLI for the convergence + per-class analysis.
    parser = argparse.ArgumentParser(description="[Step 4] Convergence analysis and per-class metric tables.")
    parser.add_argument("--saliency_dir", type=str, required=True,
                        help="Directory of per-clip artifacts saved during inference (Step 3).")
    parser.add_argument("--selected_videos_json", type=str,
                        default="train_data/json/selected_videos_for_annotations.json",
                        help="JSON mapping video_id -> event caption (for per-class grouping).")
    parser.add_argument("--tier", type=str, default="small_large_visual_cues", choices=TIERS,
                        help="Cue tier whose metrics are analyzed (default: P+S+C).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write CSVs/plot (default: <saliency_dir>/analysis).")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for confidence intervals.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="CI half-width (fraction) below which a metric is 'converged'.")
    parser.add_argument("--min_n", type=int, default=5,
                        help="Minimum clip count before convergence can be declared.")
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=BOOTSTRAP_SAMPLE_SIZES,
                        help="Clip counts at which to evaluate the bootstrap CI "
                             "(default: 50 100 150 200). Sizes above the number of "
                             "available clips are skipped.")
    parser.add_argument("--load_npz", action="store_true",
                        help="Also load frame_scores arrays from the .npz files.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.saliency_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # [Step 4] Load artifacts + event-class mapping.
    event_map = load_event_class_map(args.selected_videos_json)
    records = load_clip_records(args.saliency_dir, args.tier, event_map, load_npz=args.load_npz)
    if not records:
        print(f"[Step 4] No clip artifacts found in {args.saliency_dir}. "
              f"Run inference with --saliency_save_dir first.")
        return
    print(f"[Step 4] Loaded {len(records)} clips (tier={args.tier}).")

    # [Step 4] Convergence analysis.
    result = convergence_analysis(records, args.n_bootstrap, args.seed, args.tol,
                                  sample_sizes=args.sample_sizes, min_n=args.min_n)
    print_convergence(result)
    print_sample_size_table(result)
    write_convergence_csv(result, os.path.join(output_dir, "convergence_summary.csv"))
    write_convergence_curves_csv(result, os.path.join(output_dir, "convergence_curves.csv"))
    maybe_plot_convergence(result, os.path.join(output_dir, "convergence_curves.png"))

    # [Step 4] Per-class metrics table.
    table = per_class_table(records)
    print_per_class_table(table)
    write_per_class_csv(table, os.path.join(output_dir, "per_class_metrics.csv"))

    print(f"\n[Step 4] Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
