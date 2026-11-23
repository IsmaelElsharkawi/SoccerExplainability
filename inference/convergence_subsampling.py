"""Subsampling convergence of the Table 1 metrics.

For every model x metric (one cue tier), clips are put in a seeded random order and the
bootstrap 95% CI of the mean is recomputed on the first n clips, n = 1..N. Reads the
per_clip_<resolution>.csv written by evaluate_grounding.py; needs numpy only.

    python inference/convergence_subsampling.py <per_clip.csv> <out_dir> [tier, default P+S+C]

Outputs convergence_curves_<tier>.csv (mean and CI half-width for every n) and
convergence_summary_<tier>.{json,md} (half-width at 50/100/200/500/all clips and the smallest
n after which the half-width stays at or below TOL).
"""
import collections
import csv
import json
import sys
import zlib
from multiprocessing import Pool

import numpy as np

METRICS = ["energy", "pointing", "s_iou", "s_auroc", "s_auprc", "t_iou", "t_auroc", "t_auprc"]
N_BOOTSTRAP, TOL, SEED = 1000, 0.02, 0
DATA = collections.defaultdict(lambda: collections.defaultdict(list))


def load(src, tier):
    for row in csv.DictReader(open(src, newline="")):
        if row["tier"] != tier:
            continue
        for metric in METRICS:
            value = row[metric]
            DATA[row["model"]][metric].append(float(value) if value not in ("", "nan", "NaN") else np.nan)


def curve(job):
    model, metric = job
    x = np.array(DATA[model][metric])
    x = x[~np.isnan(x)]
    rng = np.random.default_rng([SEED, zlib.crc32(f"{model}|{metric}".encode())])
    x = x[rng.permutation(len(x))]
    halfwidth, mean = np.empty(len(x)), np.empty(len(x))
    for n in range(1, len(x) + 1):
        sample = x[:n][rng.integers(0, n, size=(N_BOOTSTRAP, n))].mean(axis=1)
        low, high = np.percentile(sample, [2.5, 97.5])
        halfwidth[n - 1], mean[n - 1] = (high - low) / 2, x[:n].mean()
    above = np.nonzero(halfwidth > TOL)[0]
    converged = int(above[-1] + 2) if len(above) else 1
    return model, metric, halfwidth, mean, (converged if converged <= len(x) else None)


def main(src, out, tier):
    load(src, tier)
    jobs = [(model, metric) for model in DATA for metric in METRICS]
    with Pool() as pool:
        results = pool.map(curve, jobs)
    rows, summary = [], []
    for model, metric, halfwidth, mean, converged in results:
        rows += [(model, metric, n + 1, f"{mean[n]:.5f}", f"{halfwidth[n]:.5f}") for n in range(len(halfwidth))]
        at = lambda n: float(halfwidth[n - 1]) if n <= len(halfwidth) else None
        summary.append(dict(model=model, metric=metric, n_valid=len(halfwidth), mean=float(mean[-1]), hw50=at(50),
                            hw100=at(100), hw200=at(200), hw500=at(500), hw_full=float(halfwidth[-1]),
                            converged_at_n=converged))
    with open(f"{out}/convergence_curves_{tier}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "n", "mean", "ci95_halfwidth"])
        writer.writerows(rows)
    json.dump(dict(tier=tier, n_bootstrap=N_BOOTSTRAP, tol=TOL, seed=SEED, rows=summary),
              open(f"{out}/convergence_summary_{tier}.json", "w"), indent=1)
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    with open(f"{out}/convergence_summary_{tier}.md", "w") as f:
        f.write(f"Bootstrap 95% CI half-width of the mean vs. number of clips (tier {tier}, {N_BOOTSTRAP} resamples, "
                f"metrics on a 0-1 scale). converged at n = smallest n after which the half-width stays <= {TOL}.\n\n"
                "| Model | Metric | valid clips | mean | hw@50 | hw@100 | hw@200 | hw@500 | hw@all | converged at n |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in summary:
            f.write(f"| {s['model']} | {s['metric']} | {s['n_valid']} | {s['mean']:.3f} | {fmt(s['hw50'])} | "
                    f"{fmt(s['hw100'])} | {fmt(s['hw200'])} | {fmt(s['hw500'])} | {fmt(s['hw_full'])} | "
                    f"{s['converged_at_n'] or '>N'} |\n")
    for key in ("hw200", "hw500", "hw_full"):
        values = [s[key] for s in summary if s[key] is not None]
        if values:
            print(key, f"range {min(values):.4f}-{max(values):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "P+S+C")
