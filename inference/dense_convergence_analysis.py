"""[Step 4b] Dense convergence analysis: every sample size n = 1..200.

Companion to ``convergence_analysis.py``. That script evaluates the bootstrap CI
at four sample sizes (50/100/150/200), which is enough to show the curve is
shrinking but too coarse to say *where* a metric settles. This one traces every
prefix size from 1 to 200, so ``converged_at_n`` becomes a real answer rather
than a choice between four buckets.

It walks the Google Drive output root and processes every run it finds, writing
alongside the existing analysis CSVs under ``<run>/saliency/analysis/`` with a
``_full`` suffix so nothing from the coarse pass is overwritten.

Two deliberate differences from the coarse script:

  * **Convergence is sustained, not first-crossing.** With 200 points the curve
    is no longer monotone -- bootstrap noise lets a metric dip under the
    tolerance and pop back out. ``converged_at_n`` is therefore the smallest n
    beyond which the half-width stays at or below ``TOL`` for *every* larger n.
    The first (possibly spurious) crossing is reported separately as
    ``first_below_n``; a wide gap between the two means a noisy curve.

  * **Each point has its own RNG stream**, seeded from (seed, metric, n), so a
    given cell reproduces regardless of what else was computed. The coarse
    script draws sequentially from one stream, so its numbers depend on
    iteration order. Expect the n=50/100/150/200 half-widths here to differ from
    ``convergence_curves.csv`` in the third decimal; that is resampling noise,
    not a disagreement.

This file depends on nothing but numpy (and matplotlib, optionally, for the
plot). Paste the whole thing into a Colab cell after mounting Drive and it runs
-- no checkout, no sys.path surgery, no imports from the rest of the repo:

    from google.colab import drive
    drive.mount('/content/drive')

    # ... paste this file below, then in the next cell:
    results = run(verbose=False, one_per_experiment=True)
    results[('chefer_siglip', 'run_0029')]['points']['T-AP']

From a shell, once Drive is mounted:

    python inference/dense_convergence_analysis.py
"""

import csv
import glob
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Self-contained by design. Everything in this block is duplicated from
# convergence_analysis.py rather than imported, because this file has to
# survive being pasted into a bare Colab cell -- where there is no repository
# to import from and no __file__ to locate one with. The definitions are
# equivalent, so both scripts produce identical numbers; if the metric
# extraction there ever changes, mirror the change here.
# ---------------------------------------------------------------------------
METRIC_ORDER = ["Energy", "Pointing", "S-IoU", "S-AUROC", "S-AUPRC",
                "T-IoU", "T-IoU (sweep)", "T-AUC", "T-AP"]


def _spatial_metrics(summary, tier):
    gs = (summary.get("label_group_scores", {}) or {}).get(tier, {}) or {}
    return {
        "Energy": gs.get("mean_energy_inside_bbox", np.nan),
        "Pointing": gs.get("mean_pointing_accuracy", np.nan),
        "S-IoU": gs.get("mean_iou", np.nan),
        "S-AUROC": gs.get("mean_s_auroc", np.nan),
        "S-AUPRC": gs.get("mean_s_auprc", np.nan),
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
    metrics = {}
    metrics.update(_spatial_metrics(summary, tier))
    metrics.update(_temporal_metrics(summary, tier))
    return metrics


def load_clip_records(saliency_dir, tier):
    """Read every <clip>.json and assemble per-clip metric records."""
    records = []
    for path in sorted(glob.glob(os.path.join(saliency_dir, "*.json"))):
        with open(path) as f:
            meta = json.load(f)
        records.append({
            "clip_id": meta.get("clip_id",
                                os.path.splitext(os.path.basename(path))[0]),
            "event_class": meta.get("ground_truth_text") or "unknown",
            "metrics": extract_clip_metrics(meta.get("summary", {}) or {}, tier),
        })
    return records


def _bootstrap_mean_ci(values, n_bootstrap, rng, alpha=0.05, sample_size=None):
    """Return (mean, ci_low, ci_high, ci_halfwidth) for a 1-D value array.

    ``sample_size`` is the size of each bootstrap resample, defaulting to the
    number of available clips. Resamples are always drawn from all clips, so no
    subset is privileged and there is no dependence on clip ordering.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if values.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    n = values.size
    m = n if sample_size is None else int(sample_size)
    if m <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    boot_means = values[rng.integers(0, n, size=(n_bootstrap, m))].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (float(values.mean()), lo, hi, float((hi - lo) / 2.0))

# ---------------------------------------------------------------------------
# Hard-coded run configuration.
# ---------------------------------------------------------------------------
# Colab mount path for the Drive output root. Every experiment lives directly
# under here as <experiment_name>/run_XXXX/saliency/.
DRIVE_ROOT = "/content/drive/MyDrive/SoccerExplainability-output"

# Consolidated outputs covering every run, written once at the end. Set either
# to None to skip it; the per-run files under Drive are written regardless.
COMBINED_CSV = "/content/convergence.csv"
COMBINED_PLOT = "/content/convergence.png"

TIER = "small_large_visual_cues"
MAX_N = 200
N_BOOTSTRAP = 1000
SEED = 0
TOL = 0.02
MIN_N = 5


def dense_curve(records, metric_index, metric, n_bootstrap, seed, max_n):
    """Bootstrap mean/CI at every sample size 1..max_n for one metric.

    Resamples are always drawn from the full clip pool, so a point at n=30
    answers "what would the CI look like with 30 clips?" without privileging any
    particular subset or depending on clip order.
    """
    values = [rec["metrics"].get(metric, np.nan) for rec in records]
    n_avail = int(sum(1 for v in values if not np.isnan(v)))
    if n_avail == 0:
        return None, []

    # Full-sample bootstrap, reused verbatim wherever the requested size clamps
    # to n_avail so the curve endpoint and the summary row cannot disagree.
    # Seed slot 0 is the full-sample stream; per-size streams use n itself,
    # which starts at 1, so the streams never collide.
    rng_full = np.random.default_rng([seed, metric_index, 0])
    mean, lo, hi, half = _bootstrap_mean_ci(values, n_bootstrap, rng_full)
    final = {"mean": mean, "ci_low": lo, "ci_high": hi,
             "ci_halfwidth": half, "n_clips": n_avail}

    curve = []
    for n in range(1, max_n + 1):
        n_used = min(n, n_avail)
        if n_used == n_avail:
            point = (mean, lo, hi, half)
        else:
            rng_n = np.random.default_rng([seed, metric_index, n])
            point = _bootstrap_mean_ci(values, n_bootstrap, rng_n, sample_size=n_used)
        curve.append((n, n_used) + point)
    return final, curve


def convergence_points(curve, tol, min_n):
    """Return (first_below_n, sustained_n) for a half-width curve.

    ``sustained_n`` walks back from the largest n and stops at the first point
    that breaches the tolerance, so it is the smallest size from which the
    metric never again exceeds ``tol``.
    """
    first_below = None
    for n, _n_used, _mean, _lo, _hi, half in curve:
        if n >= min_n and not np.isnan(half) and half <= tol:
            first_below = n
            break

    sustained = None
    for n, _n_used, _mean, _lo, _hi, half in reversed(curve):
        if np.isnan(half) or half > tol:
            break
        if n >= min_n:
            sustained = n
    return first_below, sustained


def analyse_run(saliency_dir, n_bootstrap=N_BOOTSTRAP,
                seed=SEED, max_n=MAX_N, tol=TOL, min_n=MIN_N):
    """Dense convergence analysis for one run's saliency directory."""
    records = load_clip_records(saliency_dir, TIER)
    if not records:
        return None

    final, curves, points = {}, {}, {}
    for idx, metric in enumerate(METRIC_ORDER):
        fin, curve = dense_curve(records, idx, metric, n_bootstrap, seed, max_n)
        if fin is None:
            final[metric] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                             "ci_halfwidth": np.nan, "n_clips": 0}
            curves[metric], points[metric] = [], (None, None)
            continue
        final[metric] = fin
        curves[metric] = curve
        points[metric] = convergence_points(curve, tol, min_n)

    return {"n_records": len(records), "final": final,
            "curves": curves, "points": points}


def write_curves_csv(result, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "n_clips", "n_clips_used",
                         "mean", "ci_low", "ci_high", "ci_halfwidth"])
        for metric in METRIC_ORDER:
            for n, n_used, mean, lo, hi, half in result["curves"][metric]:
                writer.writerow([metric, n, n_used, mean, lo, hi, half])


def write_summary_csv(result, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "ci_low", "ci_high", "ci_halfwidth",
                         "first_below_n", "converged_at_n", "n_clips"])
        for metric in METRIC_ORDER:
            fin = result["final"][metric]
            first_below, sustained = result["points"][metric]
            writer.writerow([metric, fin["mean"], fin["ci_low"], fin["ci_high"],
                             fin["ci_halfwidth"], first_below, sustained,
                             fin["n_clips"]])


def maybe_plot(result, path, tol=TOL, max_n=MAX_N):
    """Half-width vs n for all metrics (skipped if matplotlib is unavailable).

    Built through the object-oriented Figure/Agg API rather than pyplot, so it
    never touches matplotlib's global backend. Calling ``matplotlib.use("Agg")``
    inside a notebook would switch the session away from the inline backend and
    silently break the caller's other plots.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception:
        print("    matplotlib not available; skipping plot.")
        return

    fig = Figure(figsize=(9, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for metric in METRIC_ORDER:
        curve = result["curves"][metric]
        if not curve:
            continue
        xs = [n for n, _, _, _, _, _ in curve]
        ys = [100.0 * half for _, _, _, _, _, half in curve]
        ax.plot(xs, ys, linewidth=1.4, label=metric)

    ax.axhline(100.0 * tol, color="0.4", linestyle="--", linewidth=1,
               label=f"tolerance ({100 * tol:.0f} pp)")
    ax.set_xlabel("Number of clips (bootstrap resample size)")
    ax.set_ylabel("Bootstrap 95% CI half-width (pp)")
    ax.set_title(f"Metric convergence, n = 1..{max_n}")
    ax.set_xticks(range(0, max_n + 1, 25))
    ax.set_xlim(1, max_n)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _short(experiment):
    """chefer_soccermaster_temporal -> soccermaster."""
    name = experiment
    for prefix in ("chefer_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("_temporal", "_spatial"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def write_combined_csv(results, path):
    """Every run, metric and sample size in one long-format table."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "run_id", "metric",
                         "n_clips", "n_clips_used", "mean", "ci_low", "ci_high",
                         "ci_halfwidth", "first_below_n", "converged_at_n"])
        for (experiment, run_id), result in sorted(results.items()):
            model = _short(experiment)
            for metric in METRIC_ORDER:
                first_below, sustained = result["points"][metric]
                for n, n_used, mean, lo, hi, half in result["curves"][metric]:
                    writer.writerow([experiment, model, run_id, metric, n, n_used,
                                     mean, lo, hi, half, first_below, sustained])


def plot_combined(results, path, tol=TOL, max_n=MAX_N):
    """One panel per metric, one line per model.

    Eight metrics times several models is too many lines for a single axes, so
    this is small multiples: each panel shares the y-scale question "how wide is
    the CI at n?" but only compares models within one metric.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception:
        print("matplotlib not available; skipping combined plot.")
        return

    # Repeat runs are identical, so plot one representative run per experiment.
    by_model = {}
    for (experiment, run_id), result in sorted(results.items()):
        by_model.setdefault(_short(experiment), result)
    if not by_model:
        return

    fig = Figure(figsize=(15, 7))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4).ravel()
    colors = {m: f"C{i}" for i, m in enumerate(sorted(by_model))}

    for ax, metric in zip(axes, METRIC_ORDER):
        for model, result in sorted(by_model.items()):
            curve = result["curves"][metric]
            if not curve:
                continue
            ax.plot([n for n, _, _, _, _, _ in curve],
                    [100.0 * half for _, _, _, _, _, half in curve],
                    linewidth=1.3, color=colors[model], label=model)
        ax.axhline(100.0 * tol, color="0.45", linestyle="--", linewidth=1)
        ax.set_title(metric, fontsize=11)
        ax.set_xlim(1, max_n)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelsize=8)

    for ax in axes[4:]:
        ax.set_xlabel("clips (n)", fontsize=9)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("95% CI half-width (pp)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(by_model) + 1,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Bootstrap CI half-width vs clip count  "
                 f"(dashed = {100 * tol:.0f} pp tolerance)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)


def print_summary(result):
    print(f"    clips: {result['n_records']}")
    header = (f"    {'metric':<16}{'mean':>9}{'half':>9}"
              f"{'first<tol':>11}{'conv@n':>9}{'n':>6}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for metric in METRIC_ORDER:
        fin = result["final"][metric]
        first_below, sustained = result["points"][metric]
        mean = "--" if np.isnan(fin["mean"]) else f"{100 * fin['mean']:.2f}"
        half = "--" if np.isnan(fin["ci_halfwidth"]) else f"{100 * fin['ci_halfwidth']:.2f}"
        print(f"    {metric:<16}{mean:>9}{half:>9}"
              f"{str(first_below or '--'):>11}{str(sustained or '--'):>9}"
              f"{fin['n_clips']:>6}")


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


def _show(path):
    """Render a saved figure inline; a no-op outside a notebook."""
    try:
        from IPython.display import Image, display
    except Exception:
        return
    try:
        display(Image(filename=path))
    except Exception:
        pass


def run(root=None, show_plots=False, only=None, verbose=True,
        one_per_experiment=False, combined_csv=COMBINED_CSV,
        combined_plot=COMBINED_PLOT, max_n=MAX_N, n_bootstrap=N_BOOTSTRAP,
        seed=SEED, tol=TOL, min_n=MIN_N):
    """Analyse every run under ``root`` and return the results.

    Designed to be called straight from a notebook cell: it prints the tables
    the CLI does, but returns ``{(experiment, run): result}`` rather than an exit
    code, so the curves stay available for further poking. Returns an empty dict
    when the root is missing or holds no runs.

    Output controls, because ten runs at full verbosity is 143 lines and ten
    embedded figures:

      ``verbose=False``       one summary line per run instead of a metric table.
      ``one_per_experiment``  analyse only the first run of each experiment.
                              Repeats are byte-identical when the pipeline is
                              deterministic, so the rest are usually duplicates
                              -- but that is an assumption, so it is off by
                              default rather than silently hiding a run that
                              disagrees.
      ``show_plots``          render each figure inline. Every run still writes
                              its PNG to disk regardless.
      ``only``                filter to experiments/runs containing this string.
    """
    root = root or DRIVE_ROOT
    if not os.path.isdir(root):
        print(f"[Step 4b] Output root not found: {root}")
        print("          If this is Colab, mount Drive first:")
        print("              from google.colab import drive")
        print("              drive.mount('/content/drive')")
        return {}

    runs = find_runs(root)
    if only:
        runs = [r for r in runs if only in r[0] or only in r[1]]
    if one_per_experiment:
        seen = set()
        runs = [r for r in runs if not (r[0] in seen or seen.add(r[0]))]
    if not runs:
        print(f"[Step 4b] No <experiment>/run_XXXX/saliency directories under {root}"
              + (f" matching {only!r}." if only else "."))
        return {}

    print(f"[Step 4b] {len(runs)} run(s) under {root}, "
          f"n = 1..{max_n}, {n_bootstrap} resamples, tier={TIER}.")

    results = {}
    for experiment, run_id, saliency_dir in runs:
        if verbose:
            print(f"\n[Step 4b] {experiment}/{run_id}")
        result = analyse_run(saliency_dir, n_bootstrap=n_bootstrap,
                             seed=seed, max_n=max_n, tol=tol, min_n=min_n)
        if result is None:
            print(f"    no clip artifacts in {saliency_dir}; skipped.")
            continue

        output_dir = os.path.join(saliency_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "convergence_curves_full.png")
        write_summary_csv(result, os.path.join(output_dir, "convergence_summary_full.csv"))
        write_curves_csv(result, os.path.join(output_dir, "convergence_curves_full.csv"))
        maybe_plot(result, plot_path, tol=tol, max_n=max_n)

        if verbose:
            print_summary(result)
            print(f"    written to {output_dir}")
        else:
            converged = sum(1 for m in METRIC_ORDER if result["points"][m][1] is not None)
            print(f"  {experiment}/{run_id}: {result['n_records']} clips, "
                  f"{converged}/{len(METRIC_ORDER)} metrics converged at "
                  f"tol={100 * tol:.1f}pp")
        if show_plots and os.path.isfile(plot_path):
            _show(plot_path)

        results[(experiment, run_id)] = result

    # Consolidated outputs: one long-format CSV and one small-multiples figure
    # covering every run, so the per-run files under Drive stay for reference
    # but nothing has to be stitched together by hand.
    if results and combined_csv:
        parent = os.path.dirname(combined_csv) or "."
        if os.path.isdir(parent):
            write_combined_csv(results, combined_csv)
            rows = len(results) * len(METRIC_ORDER) * max_n
            print(f"\n[Step 4b] Combined CSV -> {combined_csv} ({rows} rows)")
        else:
            print(f"\n[Step 4b] Skipped combined CSV: {parent} does not exist.")

    if results and combined_plot:
        parent = os.path.dirname(combined_plot) or "."
        if os.path.isdir(parent):
            plot_combined(results, combined_plot, tol=tol, max_n=max_n)
            print(f"[Step 4b] Combined plot -> {combined_plot}")
            _show(combined_plot)
        else:
            print(f"[Step 4b] Skipped combined plot: {parent} does not exist.")

    print(f"\n[Step 4b] Done: {len(results)} run(s).")
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
    # A notebook cell also reports __name__ == "__main__", but raising
    # SystemExit there surfaces as a spurious traceback, so return the results
    # into the cell's namespace instead.
    if _in_notebook():
        results = run()
    else:
        raise SystemExit(main())
