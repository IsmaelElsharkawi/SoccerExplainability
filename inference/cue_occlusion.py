"""Cue occlusion: do the models need the annotated cues? (no attribution method involved)

The cue boxes allow a direct causal test of what a model relies on. For every clip and for
two cue sets (``primary`` = primary cues only, ``all`` = primary + secondary + common):

  delete_cue    the cue boxes are removed from every frame that has them
  delete_rand   boxes of the same sizes, in the same frames, at random positions that avoid
                every annotated cue (``--random_seeds`` draws): the equal-area control
  keep_cue      everything except the cue boxes is removed (frames without a box are removed
                entirely)
  keep_rand     everything except the equal-area random boxes is removed

"Removed" pixels are replaced by the per-clip channel mean (``mean``, the fill of
perturbation_faithfulness.py) or by a Gaussian-blurred copy of the frame (``blur``); both are
run by default. The clip is re-classified after each edit.

Reading the result. If a model relies on the event-defining cues, delete_cue must hurt more
than delete_rand, and keep_cue must retain more than keep_rand. If delete_cue is no worse
than delete_rand, the prediction does not depend on the cues more than on any other region
of the same size: the evidence for shortcut use, shown without any attribution map.

Readouts per condition: top-1 prediction, correctness, agreement with the clean prediction and
the softmax probability of the ground-truth class. The summary reports them on all clips and
on the correctly classified clips, for every annotation source in the COCO file, with paired
cue-minus-random differences and 95% bootstrap intervals.

Caveat for sparsely annotated clips: only annotated frames have boxes, so on the human set
(about 13 annotated frames per clip) a cue stays visible in the frames without annotations.
The silver set is annotated on every frame.

    python inference/cue_occlusion.py --model matchvision --checkpoint_path <ckpt> \
        --config_path config/pretrain_classification_1000.py --coco_json annotations-coco-merged.json \
        --out_dir <out>/matchvision
    python inference/cue_occlusion.py --out_dir <out>/combined \
        --summarize_runs MatchVision=<out>/matchvision SoccerMaster=<out>/soccermaster

Cost: 1 + 2 cue sets x fills x (2 + 2 x seeds) forward passes per clip (33 by default).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from perturbation_faithfulness import (MODELS, add_model_args, bootstrap_mean_ci, build_predictor, iter_clips,
                                       load_config, sanitize)

CUE_SETS = {"primary": {"small label"}, "all": {"small label", "large label", "visual cue"}}
CONDITIONS = ("delete_cue", "delete_rand", "keep_cue", "keep_rand")


# ============================================================================
# Regions
# ============================================================================

def box_to_pixels(box, size):
    """Normalized [x, y, w, h] -> integer (x1, y1, x2, y2) on a size x size input (same rounding as the scorer)."""
    x, y, w, h = box
    x1, y1 = max(0, int(round(x * size))), max(0, int(round(y * size)))
    x2, y2 = min(size, int(round((x + w) * size))), min(size, int(round((y + h) * size)))
    return x1, y1, x2, y2


def paint(mask, boxes):
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def random_boxes(boxes, forbidden, size, rng, tries=50):
    """One box per input box, same width and height, placed where it overlaps `forbidden` least."""
    placed = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        best, best_overlap = None, None
        for _ in range(tries):
            nx = int(rng.integers(0, size - w + 1))
            ny = int(rng.integers(0, size - h + 1))
            overlap = int(forbidden[ny:ny + h, nx:nx + w].sum())
            if best is None or overlap < best_overlap:
                best, best_overlap = (nx, ny, nx + w, ny + h), overlap
            if overlap == 0:
                break
        placed.append(best)
    return placed


def clip_regions(coco, clip_id, num_frames, size, seeds):
    """{cue_set: {'cue': (T,S,S) bool, 'rand': [(T,S,S) bool per seed], 'frames': int}} or None without boxes."""
    per_frame = []
    for t in range(num_frames):
        annots = coco.get_annotations_for_frame(clip_id, t) or []
        per_frame.append([(str(a.get("label", "")).strip().lower(), box_to_pixels(a["bbox"], size))
                          for a in annots if not a.get("no_roi", False)])
    regions = {}
    for name, labels in CUE_SETS.items():
        cue = np.zeros((num_frames, size, size), dtype=bool)
        rand = [np.zeros((num_frames, size, size), dtype=bool) for _ in seeds]
        frames = 0
        for t, items in enumerate(per_frame):
            boxes = [b for label, b in items if label in labels]
            if not boxes:
                continue
            frames += 1
            paint(cue[t], boxes)
            forbidden = paint(np.zeros((size, size), dtype=bool), [b for _, b in items])
            for k, seed in enumerate(seeds):
                rng = np.random.default_rng([seed, t])
                paint(rand[k][t], random_boxes(boxes, forbidden, size, rng))
        if frames:
            regions[name] = {"cue": cue, "rand": rand, "frames": frames}
    return regions


# ============================================================================
# Edits
# ============================================================================

def blurred_copy(x):
    """Gaussian blur of every frame; sigma is 4% of the input side (9 px at 224, 20 px at 512)."""
    from torchvision.transforms.functional import gaussian_blur
    sigma = 0.04 * x.shape[-1]
    kernel = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(x.float(), [kernel, kernel], [sigma, sigma]).to(x.dtype)


def remove(x, region, fill, blurred):
    """Replace the pixels of `region` (T, S, S bool) in x (T, 3, S, S)."""
    mask = torch.as_tensor(region, device=x.device)[:, None]
    if fill == "mean":
        value = x.float().mean(dim=(0, 2, 3)).to(x.dtype)[None, :, None, None]
        return torch.where(mask, value, x)
    return torch.where(mask, blurred, x)


# ============================================================================
# Summary
# ============================================================================

def _rows(clips, model_filter=None):
    return [c for c in clips if model_filter is None or model_filter(c)]


def _cond_value(clip, cue_set, fill, condition, key):
    entry = clip["results"].get(cue_set, {}).get(fill)
    if entry is None:
        return float("nan")
    value = entry[condition]
    if isinstance(value, list):  # random draws
        return float(np.mean([v[key] for v in value])) if value else float("nan")
    return float(value[key])


def aggregate(clips):
    """{subset: {cue_set: {fill: {...}}}} with bootstrap CIs."""
    sources = sorted({c.get("source", "unknown") for c in clips})
    subsets = {"all clips": lambda c: True, "correct": lambda c: c["clean"]["correct"]}
    for source in sources:
        subsets[f"correct, {source}"] = lambda c, s=source: c["clean"]["correct"] and c.get("source", "unknown") == s
    out = {}
    for subset, keep in subsets.items():
        rows = [c for c in clips if keep(c)]
        out[subset] = {"n": len(rows)}
        for cue_set in CUE_SETS:
            have = [c for c in rows if cue_set in c["results"]]
            fills = sorted({f for c in have for f in c["results"][cue_set]})
            out[subset][cue_set] = {"n": len(have)}
            if not have:
                continue
            out[subset][cue_set]["clean_accuracy"] = float(np.mean([c["clean"]["correct"] for c in have]))
            out[subset][cue_set]["clean_p_target"] = float(np.mean([c["clean"]["p_target"] for c in have]))
            out[subset][cue_set]["masked_fraction_cue"] = float(np.mean([c["results"][cue_set + "_meta"]["cue_fraction"] for c in have]))
            out[subset][cue_set]["masked_fraction_rand"] = float(np.mean([c["results"][cue_set + "_meta"]["rand_fraction"] for c in have]))
            for fill in fills:
                stats = {}
                for condition in CONDITIONS:
                    for key in ("correct", "agree", "p_target"):
                        stats[f"{condition}.{key}"] = bootstrap_mean_ci(
                            [_cond_value(c, cue_set, fill, condition, key) for c in have])
                for pair, (a, b) in {"delete": ("delete_cue", "delete_rand"), "keep": ("keep_cue", "keep_rand")}.items():
                    for key in ("correct", "agree", "p_target"):
                        stats[f"{pair}.cue_minus_rand.{key}"] = bootstrap_mean_ci(
                            [_cond_value(c, cue_set, fill, a, key) - _cond_value(c, cue_set, fill, b, key) for c in have])
                out[subset][cue_set][fill] = stats
    return out


def _fmt(stat, digits=3):
    if stat is None or stat.get("n", 0) == 0:
        return "n/a"
    return f"{stat['mean']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def summary_markdown(summaries, settings):
    lines = ["# Cue occlusion", "",
             f"Settings: {', '.join(f'{k}={v}' for k, v in settings.items())}", "",
             "Accuracy after each edit, with 95% bootstrap CIs over clips. delete_cue / delete_rand remove the cue "
             "boxes / equal-area random boxes; keep_cue / keep_rand remove everything else. `cue - rand` is the paired "
             "per-clip difference: negative for delete means removing the cues hurts more than removing random regions; "
             "positive for keep means the cues alone retain more than random regions. `area` is the mean share of "
             "clip pixels inside the boxes.", ""]
    first = next(iter(summaries.values()))
    for subset in first:
        for metric, title in (("correct", "accuracy"), ("p_target", "p(ground-truth class)"), ("agree", "agreement with the clean prediction")):
            lines += [f"## {subset}: {title}", "",
                      "| Model | cues | fill | n | clean | area | delete_cue | delete_rand | cue - rand | keep_cue | keep_rand | cue - rand |",
                      "|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
            for label, summary in summaries.items():
                for cue_set in CUE_SETS:
                    block = summary.get(subset, {}).get(cue_set, {})
                    if not block.get("n"):
                        continue
                    clean = {"correct": block["clean_accuracy"], "p_target": block["clean_p_target"], "agree": 1.0}[metric]
                    for fill in [f for f in ("mean", "blur") if f in block]:
                        s = block[fill]
                        lines.append(
                            f"| {label} | {cue_set} | {fill} | {block['n']} | {clean:.3f} | {block['masked_fraction_cue']:.3f} | "
                            f"{_fmt(s[f'delete_cue.{metric}'])} | {_fmt(s[f'delete_rand.{metric}'])} | {_fmt(s[f'delete.cue_minus_rand.{metric}'])} | "
                            f"{_fmt(s[f'keep_cue.{metric}'])} | {_fmt(s[f'keep_rand.{metric}'])} | {_fmt(s[f'keep.cue_minus_rand.{metric}'])} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def load_clip_results(out_dir):
    clips = []
    clips_dir = os.path.join(out_dir, "clips")
    for name in sorted(os.listdir(clips_dir)) if os.path.isdir(clips_dir) else []:
        if name.endswith(".json"):
            with open(os.path.join(clips_dir, name)) as f:
                clips.append(json.load(f))
    return clips


def write_summary(out_dir, label, clips, settings):
    summary = aggregate(clips)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"label": label, "settings": settings, "summary": summary}, f, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown({label: summary}, settings))
    return summary


# ============================================================================
# Driver
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--config_path", help="Same config as the attribution scripts (test split + keywords).")
    parser.add_argument("--coco_json", help="COCO file with the cue boxes.")
    parser.add_argument("--out_dir")
    parser.add_argument("--label", default=None)
    parser.add_argument("--fills", nargs="+", default=["mean", "blur"], choices=("mean", "blur"))
    parser.add_argument("--random_seeds", type=int, default=3, help="Equal-area random placements per clip.")
    parser.add_argument("--logit_scale", type=float, default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--summarize_runs", nargs="*", default=None, metavar="LABEL=OUT_DIR")
    add_model_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.out_dir:
        raise SystemExit("--out_dir is required")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.summarize_runs is not None:
        summaries, settings = {}, {}
        for spec in args.summarize_runs:
            label, run_dir = spec.split("=", 1)
            clips = load_clip_results(run_dir)
            if not clips:
                raise SystemExit(f"no per-clip results under {run_dir}/clips")
            settings = clips[0].get("settings", {})
            summaries[label] = aggregate(clips)
        with open(os.path.join(args.out_dir, "summary_all.md"), "w") as f:
            f.write(summary_markdown(summaries, settings))
        with open(os.path.join(args.out_dir, "summary_all.json"), "w") as f:
            json.dump({"settings": settings, "summaries": summaries}, f, indent=1)
        print(f"Wrote {os.path.join(args.out_dir, 'summary_all.md')}")
        return

    label = args.label or args.model
    settings = {"fills": list(args.fills), "random_seeds": args.random_seeds}
    if args.summarize_only:
        clips = load_clip_results(args.out_dir)
        print(summary_markdown({label: write_summary(args.out_dir, label, clips, clips[0].get("settings", settings))}, settings))
        return
    for required in ("model", "config_path", "coco_json"):
        if getattr(args, required) is None:
            raise SystemExit(f"--{required} is required")

    from coco_attribution_eval import CocoAttributionEvaluator
    from dataset.video_utils_siglip import read_frames_decord

    config = load_config(args.config_path)
    test_config = config["dataset"]["test"]
    keywords = test_config["keywords"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = CocoAttributionEvaluator(args.coco_json)
    source_by_video = {img["video_id"]: img.get("source", "unknown") for img in coco.coco["images"]}
    predictor = build_predictor(args, keywords, device, config)
    logit_scale = args.logit_scale if args.logit_scale is not None else predictor.logit_scale
    settings.update({"logit_scale": logit_scale, "model": args.model})
    print(f"[{label}] {1 + len(CUE_SETS) * len(args.fills) * (2 + 2 * args.random_seeds)} forwards per clip")

    def classify(inp, target, pred_clean=None):
        logits = predictor.forward(inp).detach().float().cpu()
        probs = torch.softmax(logit_scale * logits, dim=-1).numpy()
        pred = int(logits.argmax())
        return {"pred": pred, "correct": bool(pred == target), "p_target": float(probs[target]),
                "agree": bool(pred == pred_clean) if pred_clean is not None else True}

    clips_dir = os.path.join(args.out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    started, n_done = time.time(), 0
    for index, (clip_id, video_path, caption) in enumerate(iter_clips(test_config)):
        if args.max_clips is not None and index >= args.max_clips:
            break
        result_path = os.path.join(clips_dir, sanitize(clip_id) + ".json")
        if not args.overwrite and os.path.isfile(result_path):
            continue
        target = keywords.index(caption)
        frames, frame_indices, _ = read_frames_decord(video_path, args.num_frames, test_config.get("sample", "middle"))
        with torch.no_grad():
            x = predictor.preprocess(frames)
            size = x.shape[-1]
            seeds = [1000 * index + s for s in range(args.random_seeds)]
            regions = clip_regions(coco, clip_id, x.shape[0], size, seeds)
            clean = classify(x, target)
            blurred = blurred_copy(x) if "blur" in args.fills else None
            results = {}
            for cue_set, region in regions.items():
                results[cue_set + "_meta"] = {"frames_with_boxes": region["frames"],
                                              "cue_fraction": float(region["cue"].mean()),
                                              "rand_fraction": float(np.mean([r.mean() for r in region["rand"]]))}
                results[cue_set] = {}
                for fill in args.fills:
                    run = lambda reg: classify(remove(x, reg, fill, blurred), target, clean["pred"])
                    results[cue_set][fill] = {
                        "delete_cue": run(region["cue"]),
                        "delete_rand": [run(r) for r in region["rand"]],
                        "keep_cue": run(~region["cue"]),
                        "keep_rand": [run(~r) for r in region["rand"]],
                    }
        record = {"clip_id": clip_id, "source": source_by_video.get(clip_id, "unknown"), "target_index": target,
                  "target_text": caption, "clean": clean, "results": results, "settings": settings}
        with open(result_path, "w") as f:
            json.dump(record, f)
        n_done += 1
        note = ""
        if "primary" in results:
            first = results["primary"][args.fills[0]]
            note = (f" primary[{args.fills[0]}]: del_cue p={first['delete_cue']['p_target']:.3f} "
                    f"del_rand p={np.mean([r['p_target'] for r in first['delete_rand']]):.3f} "
                    f"keep_cue p={first['keep_cue']['p_target']:.3f}")
        print(f"  [{index + 1}] {clip_id}: target={caption!r} pred={keywords[clean['pred']]!r} p={clean['p_target']:.3f}"
              f"{note} ({time.time() - started:.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clips = load_clip_results(args.out_dir)
    summary = write_summary(args.out_dir, label, clips, settings)
    print(f"[{label}] scored {n_done} new clips, {len(clips)} total -> {args.out_dir}/summary.md")
    print(summary_markdown({label: summary}, settings))


if __name__ == "__main__":
    main()
