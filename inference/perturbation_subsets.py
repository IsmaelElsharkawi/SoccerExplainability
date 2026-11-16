"""Subset breakdown of finished perturbation runs (no model needed).

Reads the per-clip results written by perturbation_faithfulness.py and reports, on the
correctly classified clips: all clips, the human-annotated clips, the remaining (silver)
clips and, for runs with empty maps (e.g. Grad-CAM), the clips whose map is not constant.

    python inference/perturbation_subsets.py <human_clip_list.json> "Label=<out_dir>" ["Label=<out_dir>" ...]

<human_clip_list.json> is train_data/json/selected_videos_for_annotations.json.
"""
import glob
import json
import sys

import numpy as np


def describe(rows):
    if not rows:
        return "n=0"
    mean = lambda f: float(np.mean([f(r) for r in rows]))
    rand = lambda r, *keys: np.mean([x[keys[0]][keys[1]] if len(keys) > 1 else x[keys[0]] for x in r["random"]])
    return ("n=%4d SaCo %.3f (rand %+.3f) | AUC MoRF %.3f rand %.3f LeRF %.3f | first flip MoRF %.2f rand %.2f LeRF %.2f" % (
        len(rows), mean(lambda r: r["saliency"]["saco"]), mean(lambda r: rand(r, "saco")),
        mean(lambda r: r["saliency"]["morf"]["auc"]), mean(lambda r: rand(r, "morf", "auc")),
        mean(lambda r: r["saliency"]["lerf"]["auc"]), mean(lambda r: r["saliency"]["morf"]["first_flip_fraction"]),
        mean(lambda r: rand(r, "morf", "first_flip_fraction")), mean(lambda r: r["saliency"]["lerf"]["first_flip_fraction"])))


def main(human_json, specs):
    human = {clip["video"] for clip in json.load(open(human_json))}
    is_human = lambda r: r["clip_id"].split("#")[0] in human
    empty = lambda r: np.ptp(r["saliency"]["group_salience"]) < 1e-12
    for spec in specs:
        label, run_dir = spec.split("=", 1)
        rows = [json.load(open(f)) for f in glob.glob(f"{run_dir}/clips/*.json")]
        correct = [r for r in rows if r["pred_clean"] == r["target_index"]]
        n_empty = sum(empty(r) for r in rows)
        print(f"## {label} ({len(rows)} clips, {len(correct)} correct, {n_empty} empty maps)")
        print(f"  {'correct, all':22s} {describe(correct)}")
        print(f"  {'correct, human':22s} {describe([r for r in correct if is_human(r)])}")
        print(f"  {'correct, silver':22s} {describe([r for r in correct if not is_human(r)])}")
        if n_empty:
            print(f"  {'correct, non-empty':22s} {describe([r for r in correct if not empty(r)])}")
            print(f"  {'all, non-empty':22s} {describe([r for r in rows if not empty(r)])}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
