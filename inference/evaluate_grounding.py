"""Build Table 1 (grounding results) from saved saliency maps.

Scores any number of runs with the same code (`grounding_metrics.py`) and
writes, per scoring resolution, the per-clip rows, a JSON summary, Table 1 as
Markdown and LaTeX, and a validation report against the values currently in
the paper.

Each run is a directory of `<clip>.npz` / `<clip>.json` pairs as written by
any attribution script's `--saliency_save_dir`. Pass them as LABEL=DIR, in
the order the rows should appear:

    python evaluate_grounding.py \
        --coco_json ../annotations-coco.json \
        --run SigLIP=/runs/siglip/saliency \
        --run CLIP=/runs/clip/saliency \
        --run PE-AV=/runs/peav/saliency \
        --run VideoPrism=/runs/videoprism/saliency \
        --run MatchVision=/runs/matchvision/saliency \
        --run SoccerMaster=/runs/soccermaster/saliency \
        --resolution frame native \
        --out_dir /runs/table1

`frame` (bilinear upsampling to a 720x1280 canvas) is the headline. `native`
scores on each model's token grid, as the inference scripts always have, and
is what the validation report should be read against.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grounding_metrics import (  # noqa: E402
    DEFAULT_CANVAS,
    SPATIAL_METRICS,
    TIER_SHORT,
    TIERS,
    GroundingScorer,
    aggregate,
    load_saliency_dir,
)

TABLE_METRICS = [
    ("energy", "Energy"),
    ("pointing", "Pointing"),
    ("s_auroc", "S-AUROC"),
    ("s_auprc", "S-AUPRC"),
    ("t_iou", "T-IoU"),
]
# Tolerance (percentage points) for "reproduces the paper" in the validation report.
VALIDATION_TOLERANCE = 0.05

_SCORER = None


def _init_worker(coco_json, resolution, canvas):
    global _SCORER
    _SCORER = GroundingScorer(coco_json, resolution=resolution, canvas=canvas)


def _score(args):
    clip_id, maps = args
    return clip_id, _SCORER.score_clip(clip_id, maps)


def _normalize_label(text):
    return str(text or "").strip().lower()


def load_run(label, saliency_dir, map_key, annotated_ids):
    """Read a run and run the provenance checks that have caught bad bundles
    before: missing clips, a collapsed classifier, unannotated clip ids."""
    if not os.path.isdir(saliency_dir):
        raise FileNotFoundError(f"[{label}] saliency dir not found: {saliency_dir}")

    clips, correct, predicted, warnings = [], 0, [], []
    digest = hashlib.md5()
    for clip_id, maps, meta in load_saliency_dir(saliency_dir, map_key):
        if clip_id not in annotated_ids:
            warnings.append(f"clip not in annotations, skipped: {clip_id}")
            continue
        clips.append((clip_id, maps))
        digest.update(clip_id.encode())
        digest.update(np.ascontiguousarray(maps, dtype=np.float32).tobytes())
        pred, gt = meta.get("prediction_text"), meta.get("ground_truth_text")
        if pred is not None and gt is not None:
            predicted.append(_normalize_label(pred))
            correct += _normalize_label(pred) == _normalize_label(gt)

    found = {c for c, _ in clips}
    missing = sorted(annotated_ids - found)
    if missing:
        warnings.append(f"{len(missing)} annotated clips have no saliency artifact, e.g. {missing[:3]}")
    if len(predicted) < len(clips):
        warnings.append(f"{len(clips) - len(predicted)} clips have no prediction_text/ground_truth_text")
    distinct = len(set(predicted))
    if predicted and distinct <= 2:
        warnings.append(f"only {distinct} distinct prediction(s) across {len(predicted)} clips: collapsed model?")

    grids = sorted({tuple(m.shape[1:]) for _, m in clips})
    return {
        "label": label,
        "saliency_dir": saliency_dir,
        "clips": clips,
        "accuracy": correct / len(predicted) if predicted else float("nan"),
        "n_predictions": len(predicted),
        "n_distinct_predictions": distinct,
        "maps_md5": digest.hexdigest(),
        "native_grids": [list(g) for g in grids],
        "warnings": warnings,
    }


def score_run(run, coco_json, resolution, canvas, workers):
    jobs = run["clips"]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(coco_json, resolution, canvas)) as pool:
            results = list(pool.map(_score, jobs, chunksize=4))
    else:
        _init_worker(coco_json, resolution, canvas)
        results = [_score(job) for job in jobs]
    return dict(results)


def chance_row(per_clip_by_run):
    """S-AUPRC chance is the cue-box share of the frame, aggregated like the
    metric itself; it depends only on the annotations and the resolution, so
    any run with all clips gives the same value."""
    best = max(per_clip_by_run.values(), key=len)
    summary = aggregate(list(best.values()))
    return {tier: summary[tier]["box_fraction"] for tier in TIERS}


def fmt(value, digits=2):
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def write_per_clip_csv(path, per_clip_by_run):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        cols = list(SPATIAL_METRICS) + ["t_iou", "n_frames", "n_gt_frames", "n_pred_frames"]
        writer.writerow(["model", "clip_id", "tier"] + cols)
        for label, per_clip in per_clip_by_run.items():
            for clip_id, result in sorted(per_clip.items()):
                for tier in TIERS:
                    writer.writerow([label, clip_id, TIER_SHORT[tier]] + [result[tier][c] for c in cols])


def table_markdown(summaries, accuracies, chance, resolution, canvas, image_models):
    tiers = [TIER_SHORT[t] for t in TIERS]
    header = ["Model", "Acc."] + [f"{name} {t}" for _, name in TABLE_METRICS for t in tiers]
    lines = [
        f"Table 1 recomputed ({resolution} resolution"
        + (f", canvas {canvas[0]}x{canvas[1]}" if resolution == "frame" else "")
        + "). Values in %. † = image model, standard Chefer.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    chance_cells = ["Chance", "–"]
    for key, _ in TABLE_METRICS:
        for tier in TIERS:
            chance_cells.append(fmt(50.0) if key == "s_auroc" else fmt(100 * chance[tier]) if key == "s_auprc" else "–")
    lines.append("| " + " | ".join(chance_cells) + " |")
    for label, summary in summaries.items():
        name = label + ("†" if label in image_models else "")
        cells = [name, fmt(100 * accuracies[label], 1)]
        for key, _ in TABLE_METRICS:
            cells += [fmt(100 * summary[tier][key]) for tier in TIERS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_latex(summaries, accuracies, chance, image_models):
    columns = [(key, tier) for key, _ in TABLE_METRICS for tier in TIERS]
    best = {}
    for column in columns:
        values = [summaries[m][column[1]][column[0]] for m in summaries]
        finite = [v for v in values if not np.isnan(v)]
        best[column] = max(finite) if finite else None
    acc_values = [a for a in accuracies.values() if not np.isnan(a)]
    best_acc = max(acc_values) if acc_values else None

    def cell(value, best_value, digits=2, scale=100):
        if value is None or np.isnan(value):
            return "--"
        text = f"{scale * value:.{digits}f}"
        return rf"\textbf{{{text}}}" if best_value is not None and np.isclose(value, best_value) else text

    lines = [
        r"\begin{tabular}{l c " + " ".join(["ccc"] * len(TABLE_METRICS)) + "}",
        r"\toprule",
        " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in TABLE_METRICS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(TABLE_METRICS))),
        r"Model & Acc. & " + " & ".join(["P & P+S & P+S+C"] * len(TABLE_METRICS)) + r" \\",
        r"\midrule",
    ]
    chance_cells = []
    for key, _ in TABLE_METRICS:
        for tier in TIERS:
            chance_cells.append("50.0" if key == "s_auroc" else f"{100 * chance[tier]:.1f}" if key == "s_auprc" else "--")
    lines.append("Chance & -- & " + " & ".join(chance_cells) + r" \\")
    lines.append(r"\midrule")
    for label, summary in summaries.items():
        name = label + (r"$^\dagger$" if label in image_models else "")
        cells = [cell(summary[tier][key], best[(key, tier)]) for key, tier in columns]
        lines.append(f"{name} & {cell(accuracies[label], best_acc, digits=1)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def validation_report(summaries, accuracies, chance, reference, resolution):
    """Cell-by-cell comparison with the paper's current Table 1."""
    lines = [
        f"Validation against the paper's Table 1 ({resolution} resolution). "
        f"Differences in percentage points; |Δ| ≤ {VALIDATION_TOLERANCE} counts as reproduced.",
    ]
    if resolution != "native":
        lines.append("The paper's numbers were scored on the native token grid, so deltas here "
                     "measure the effect of upsampling, not pipeline errors.")
    lines.append("")
    lines.append("| Model | Metric | Tier | Paper | Recomputed | Δ |")
    lines.append("|---|---|---|---|---|---|")
    n_checked = n_ok = 0
    for label, summary in summaries.items():
        ref = reference.get("models", {}).get(label)
        if ref is None:
            lines.append(f"| {label} | – | – | not in reference | – | – |")
            continue
        pairs = [("accuracy", None, ref.get("accuracy"), 100 * accuracies[label])]
        for key, _ in TABLE_METRICS:
            for i, tier in enumerate(TIERS):
                paper = (ref.get(key) or [None] * 3)[i]
                pairs.append((key, TIER_SHORT[tier], paper, 100 * summary[tier][key]))
        for key, tier, paper, ours in pairs:
            if paper is None:
                lines.append(f"| {label} | {key} | {tier or ''} | – | {fmt(ours)} | new |")
                continue
            if np.isnan(ours):
                lines.append(f"| {label} | {key} | {tier or ''} | {paper:.2f} | missing | – |")
                continue
            delta = ours - paper
            n_checked += 1
            ok = abs(delta) <= (VALIDATION_TOLERANCE if key != "accuracy" else 0.01) + 1e-9
            n_ok += ok
            lines.append(f"| {label} | {key} | {tier or ''} | {paper:.2f} | {ours:.2f} | "
                         f"{'' if ok else '**'}{delta:+.2f}{'' if ok else '**'} |")
    for i, tier in enumerate(TIERS):
        paper = reference.get("chance", {}).get("s_auprc", [None] * 3)[i]
        if paper is not None:
            delta = 100 * chance[tier] - paper
            lines.append(f"| Chance | s_auprc | {TIER_SHORT[tier]} | {paper:.1f} | {100 * chance[tier]:.2f} | {delta:+.2f} |")
    lines.insert(1, f"Reproduced {n_ok}/{n_checked} paper cells.")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="Row label and saliency directory; repeat in table order.")
    parser.add_argument("--resolution", nargs="+", default=["frame", "native"], choices=["frame", "native"])
    parser.add_argument("--canvas", default=f"{DEFAULT_CANVAS[0]}x{DEFAULT_CANVAS[1]}",
                        help="HEIGHTxWIDTH of the frame canvas maps are upsampled to (default: 720x1280).")
    parser.add_argument("--map_key", default="saliency_maps",
                        help="npz array to score (default: saliency_maps, i.e. Chefer-T for video models).")
    parser.add_argument("--image_models", default="SigLIP,CLIP",
                        help="Labels marked with a dagger (no temporal attention, standard Chefer).")
    parser.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "table1_reference.json"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    canvas = tuple(int(v) for v in args.canvas.lower().split("x"))
    image_models = {m.strip() for m in args.image_models.split(",") if m.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    annotated_ids = GroundingScorer(args.coco_json, resolution="native").annotated_clip_ids()
    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--run expects LABEL=DIR, got {spec!r}")
        run = load_run(label.strip(), path.strip(), args.map_key, annotated_ids)
        print(f"[{run['label']}] {len(run['clips'])} clips, accuracy {100 * run['accuracy']:.1f}% "
              f"({run['n_distinct_predictions']} distinct predictions), grids {run['native_grids']}")
        for warning in run["warnings"]:
            print(f"  WARNING: {warning}")
        runs.append(run)

    by_md5 = {}
    for run in runs:
        by_md5.setdefault(run["maps_md5"], []).append(run["label"])
    for labels in by_md5.values():
        if len(labels) > 1:
            print(f"WARNING: runs {labels} have byte-identical saliency maps (duplicated bundle?)")

    reference = {}
    if args.reference and os.path.isfile(args.reference):
        with open(args.reference) as f:
            reference = json.load(f)

    accuracies = {run["label"]: run["accuracy"] for run in runs}
    provenance = {run["label"]: {k: run[k] for k in ("saliency_dir", "maps_md5", "native_grids",
                                                      "n_predictions", "n_distinct_predictions", "warnings")}
                  for run in runs}

    for resolution in args.resolution:
        print(f"\n=== Scoring at {resolution} resolution ===")
        per_clip_by_run, summaries = {}, {}
        for run in runs:
            per_clip = score_run(run, args.coco_json, resolution, canvas, args.workers)
            per_clip_by_run[run["label"]] = per_clip
            summaries[run["label"]] = aggregate(list(per_clip.values()))
            print(f"  {run['label']}: done ({len(per_clip)} clips)")
        chance = chance_row(per_clip_by_run)

        suffix = resolution
        write_per_clip_csv(os.path.join(args.out_dir, f"per_clip_{suffix}.csv"), per_clip_by_run)
        markdown = table_markdown(summaries, accuracies, chance, resolution, canvas, image_models)
        with open(os.path.join(args.out_dir, f"table1_{suffix}.md"), "w") as f:
            f.write(markdown)
        with open(os.path.join(args.out_dir, f"table1_{suffix}.tex"), "w") as f:
            f.write(table_latex(summaries, accuracies, chance, image_models))
        with open(os.path.join(args.out_dir, f"summary_{suffix}.json"), "w") as f:
            json.dump({
                "resolution": resolution,
                "canvas": list(canvas) if resolution == "frame" else None,
                "map_key": args.map_key,
                "accuracy": accuracies,
                "chance_box_fraction": chance,
                "summary": summaries,
                "provenance": provenance,
            }, f, indent=2)
        if reference:
            report = validation_report(summaries, accuracies, chance, reference, resolution)
            with open(os.path.join(args.out_dir, f"validation_{suffix}.md"), "w") as f:
                f.write(report)
            print(report.splitlines()[1])
        print()
        print(markdown)

    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
