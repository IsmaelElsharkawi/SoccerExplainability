"""Relevance-guided frame selection: does the model need all 30 frames?

A practical corollary of the perturbation test. Every saved saliency volume
carries a per-frame relevance r_t = sum of the map over frame t (the
``frame_scores`` of a Table 1 npz; for the Chefer maps this is the temporal
weight times the spatial mass). If the maps are faithful, the k most relevant
frames should be *sufficient*: feeding only them should reproduce the
full-clip decision, and at a smaller k than the uniform sampling every video
pipeline uses by default.

Per clip and frame budget k (``--budgets``, default 1 3 5 10 of 30)
the following k-frame subsets are scored:

  relevance   the k frames with the largest r_t                (the maps)
  uniform     k evenly spaced frames (what a k-frame sampler takes)
  random      k random frames (``--random_seeds`` draws)
  least       the k frames with the smallest r_t               (negative control)
  transfer    top-k by *another* model's r_t (``--rank_saliency_dir``):
              a cheap model selecting frames for an expensive one

The frames outside the subset are removed in one of two ways:

  --mode mask  they stay in the input with every patch filled by the per-clip
               channel mean: the same operation as perturbation_faithfulness
               (``--fill mean``). Works for every model and keeps the temporal
               positions, so it is the controlled comparison.
  --mode drop  they are removed and the model sees k frames: the real compute
               saving. Only for models without a fixed-length temporal
               embedding (SigLIP and CLIP here; MatchVision, SoccerMaster,
               PE-AV and VideoPrism expect a fixed number of frames).

Readout per (condition, k): top-1 prediction, its agreement with the
full-clip prediction, correctness w.r.t. the ground truth, and the softmax
probability of the full-clip predicted class. Per condition the summary
reports agreement and accuracy vs k with bootstrap CIs, the mean agreement
over the budgets, the smallest budget whose mean agreement reaches 90%, and
the per-clip minimal sufficient k (smallest budget that reproduces the
full-clip prediction; T when none does).

Label leakage. The saved maps explain the *ground-truth* class. On a
misclassified clip, choosing frames by such a map injects label information
that a deployed selector would not have, so the primary subset is "correct"
(full-clip prediction == ground truth: there the explained class is the
predicted class). The "all clips" table is kept for completeness; its uniform
and random rows are leakage-free, its relevance/least/transfer rows are not.
With ``--rank_saliency_dir`` a third subset "correct_both" also requires the
ranking model's saved prediction to be correct.

Outputs
-------
    <out_dir>/clips/<stem>.json   per-clip predictions per condition and budget (resumable)
    <out_dir>/summary.json        aggregates per subset with bootstrap CIs
    <out_dir>/summary.md          the same as tables
    <out_dir>/curves.csv          agreement / accuracy per condition and budget
    <out_dir>/curves.png          agreement vs frames kept

``--summarize_runs Label=<out_dir> ...`` combines finished runs (table + figure).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perturbation_faithfulness import (  # noqa: E402
    MODELS, add_model_args, bootstrap_mean_ci, build_predictor, iter_clips, load_config, mask_tokens, sanitize)

DEFAULT_BUDGETS = (1, 3, 5, 10)
CONDITIONS = ("relevance", "transfer", "uniform", "random", "least")
FRAME_INDEPENDENT_MODELS = ("siglip", "clip")  # mean-pooled frame embeddings: accept any number of frames
SUBSET_ORDER = ("correct", "correct_both", "all")
SUBSET_TITLES = {
    "correct": "Correctly classified clips (primary: explained class == predicted class)",
    "correct_both": "Clips classified correctly by both the evaluated and the ranking model",
    "all": "All clips (relevance / least / transfer rows leak the label on misclassified clips)",
}
CONDITION_TEXT = {
    "relevance": "top-k frames by the model's own saved map",
    "transfer": "top-k frames by another model's saved map",
    "uniform": "k evenly spaced frames (default sampler)",
    "random": "k random frames (mean over draws)",
    "least": "bottom-k frames by the saved map (control)",
}
# Fixed-order categorical palette (validated for CVD separation); line styles
# and markers are the secondary encoding.
PLOT_STYLE = {
    "relevance": ("#2a78d6", "-", "o"),
    "uniform": ("#eb6834", "--", "s"),
    "random": ("#1baf7a", ":", "^"),
    "least": ("#eda100", "-.", "v"),
    "transfer": ("#e87ba4", "-", "D"),
}


# ============================================================================
# Frame selection (model-agnostic)
# ============================================================================

def frame_relevance(saliency):
    """r_t: summed salience per frame of a (T, h, w) volume."""
    saliency = np.asarray(saliency, dtype=np.float64)
    return saliency.reshape(saliency.shape[0], -1).sum(axis=1)


def uniform_frames(num_frames, k):
    """Middle frame of each of k equal segments: the 'middle' sampler at k frames."""
    return np.floor((np.arange(k) + 0.5) * num_frames / k).astype(int)


def select_frames(relevance, k, condition, rng=None, other=None):
    """Sorted indices of the k frames kept under ``condition``."""
    relevance = np.asarray(relevance, dtype=np.float64)
    num_frames = len(relevance)
    if not 1 <= k <= num_frames:
        raise ValueError(f"budget {k} outside 1..{num_frames}")
    if condition == "relevance":
        idx = np.argsort(-relevance, kind="stable")[:k]
    elif condition == "least":
        idx = np.argsort(relevance, kind="stable")[:k]
    elif condition == "uniform":
        idx = uniform_frames(num_frames, k)
    elif condition == "random":
        idx = rng.choice(num_frames, size=k, replace=False)
    elif condition == "transfer":
        if other is None or len(other) != num_frames:
            raise ValueError("transfer needs the other model's relevance of the same length")
        idx = np.argsort(-np.asarray(other, dtype=np.float64), kind="stable")[:k]
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return np.sort(idx)


def restrict_frames(x, keep, grid, mode="mask", fill="mean"):
    """Input (T, 3, S, S) reduced to the frames ``keep``: masked (T frames) or dropped (k frames)."""
    keep = np.asarray(keep, dtype=np.int64)
    if mode == "drop":
        return x[torch.as_tensor(keep, device=x.device)]
    if mode != "mask":
        raise ValueError(f"unknown mode {mode!r}")
    token_mask = np.ones((x.shape[0], grid, grid), dtype=bool)
    token_mask[keep] = False
    return mask_tokens(x, token_mask, fill)


# ============================================================================
# Scoring
# ============================================================================

def score_clip(predictor, x, relevance, target, budgets, mode, fill, seeds, logit_scale, other=None):
    """Full-clip prediction plus every (condition, budget) prediction for one clip."""
    def probs(inp):
        logits = predictor.forward(inp).detach().float().cpu()
        return torch.softmax(logit_scale * logits, dim=-1).numpy(), int(logits.argmax())

    num_frames = x.shape[0]
    conditions = ["relevance", "uniform", "least"] + (["transfer"] if other is not None else [])
    with torch.no_grad():
        p_clean, pred_clean = probs(x)

        def entry(k, keep, p, pred):
            return {"k": int(k), "frames": [int(i) for i in keep], "pred": int(pred),
                    "agree": bool(pred == pred_clean), "correct": bool(pred == target),
                    "p_pred_clean": float(p[pred_clean]), "p_target": float(p[target])}

        result = {c: [] for c in conditions}
        result["random"] = [[] for _ in seeds]
        for k in budgets:
            for c in conditions:
                keep = select_frames(relevance, k, c, other=other)
                p, pred = probs(restrict_frames(x, keep, predictor.grid, mode, fill))
                result[c].append(entry(k, keep, p, pred))
            for s, seed in enumerate(seeds):
                keep = select_frames(relevance, k, "random", rng=np.random.default_rng(seed))
                p, pred = probs(restrict_frames(x, keep, predictor.grid, mode, fill))
                result["random"][s].append(entry(k, keep, p, pred))

    result.update({
        "p_clean": p_clean.tolist(),
        "p_clean_target": float(p_clean[target]),
        "pred_clean": pred_clean,
        "num_frames": int(num_frames),
        "budgets": [int(k) for k in budgets],
        "frame_relevance": [float(r) for r in relevance],
        "random_seeds": [int(s) for s in seeds],
    })
    return result


# ============================================================================
# Aggregation
# ============================================================================

def _curves(clip, condition):
    """Per-clip (agree, correct, p_pred_clean) arrays over the budgets; random averaged over draws."""
    lists = clip["random"] if condition == "random" else ([clip[condition]] if condition in clip else [])
    if not lists:
        return None
    agree = np.mean([[float(e["agree"]) for e in entries] for entries in lists], axis=0)
    correct = np.mean([[float(e["correct"]) for e in entries] for entries in lists], axis=0)
    p = np.mean([[e["p_pred_clean"] for e in entries] for entries in lists], axis=0)
    return agree, correct, p


def minimal_sufficient_k(clip, condition):
    """Smallest budget whose prediction equals the full-clip one (T if none); random: mean over draws."""
    lists = clip["random"] if condition == "random" else [clip[condition]]
    ks = [next((e["k"] for e in entries if e["agree"]), clip["num_frames"]) for entries in lists]
    return float(np.mean(ks))


def aggregate(clips):
    """{subset: {...}} with per-condition agreement / accuracy curves and scalars."""
    def correct(c):
        return c["pred_clean"] == c["target_index"]

    subsets = {"all": clips, "correct": [c for c in clips if correct(c)]}
    if any("transfer_saved_correct" in c for c in clips):
        subsets["correct_both"] = [c for c in clips if correct(c) and c.get("transfer_saved_correct")]

    out = {}
    for name, rows in subsets.items():
        if not rows:
            out[name] = {"n": 0}
            continue
        num_frames = rows[0]["num_frames"]
        budgets = list(rows[0]["budgets"])
        budgets_full = budgets + [num_frames]
        clean_acc = float(np.mean([correct(c) for c in rows]))
        conditions = {}
        for cond in CONDITIONS:
            curves = [_curves(c, cond) for c in rows]
            if any(cv is None for cv in curves):
                continue
            agree = np.array([cv[0] for cv in curves])      # (n, B)
            acc = np.array([cv[1] for cv in curves])
            p = np.array([cv[2] for cv in curves])
            agree_ci = [bootstrap_mean_ci(agree[:, b]) for b in range(len(budgets))]
            agree_ci.append({"mean": 1.0, "std": 0.0, "ci_low": 1.0, "ci_high": 1.0, "n": len(rows)})
            acc_ci = [bootstrap_mean_ci(acc[:, b]) for b in range(len(budgets))]
            acc_ci.append(bootstrap_mean_ci([float(correct(c)) for c in rows]))
            means = [s["mean"] for s in agree_ci]
            k90 = next((k for k, m in zip(budgets_full, means) if m >= 0.9), num_frames)
            conditions[cond] = {
                "agreement": agree_ci,
                "accuracy": acc_ci,
                "p_pred_clean": [float(v) for v in p.mean(axis=0)] + [
                    float(np.mean([c["p_clean"][c["pred_clean"]] for c in rows]))],
                "mean_agreement": bootstrap_mean_ci(agree.mean(axis=1)),
                "k90": int(k90),
                "min_sufficient_k": bootstrap_mean_ci([minimal_sufficient_k(c, cond) for c in rows]),
            }
        out[name] = {"n": len(rows), "num_frames": num_frames, "budgets": budgets_full,
                     "clean_accuracy": clean_acc, "conditions": conditions}
    return out


# ============================================================================
# Reports
# ============================================================================

def _fmt(stat, digits=3):
    if stat is None or stat.get("n", 0) == 0 or np.isnan(stat["mean"]):
        return "n/a"
    return f"{stat['mean']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def summary_markdown(summaries, settings):
    lines = ["# Relevance-guided frame selection", ""]
    lines.append("Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    lines.append("")
    lines.append("For each frame budget k only k of the T frames are kept (the others are "
                 f"{'removed' if settings.get('mode') == 'drop' else 'filled with the per-clip channel mean'}) "
                 "and the clip is re-classified. Agreement = top-1 prediction equals the full-clip prediction; "
                 "accuracy = equals the ground truth. Values are means over clips with 95% bootstrap CIs. "
                 "k@90% = smallest budget whose mean agreement reaches 90%. Minimal sufficient frames = per-clip "
                 "smallest budget that reproduces the full-clip prediction (T if none).")
    lines.append("")
    lines.append("Conditions: " + "; ".join(f"**{c}** = {CONDITION_TEXT[c]}" for c in CONDITIONS))
    lines.append("")
    for subset in SUBSET_ORDER:
        if not any(subset in agg for agg in summaries.values()):
            continue
        lines.append(f"## {SUBSET_TITLES[subset]}")
        lines.append("")
        lines.append("| Model | condition | n | clean accuracy | mean agreement over budgets | k@90% | "
                     "minimal sufficient frames |")
        lines.append("|---|---|---:|---:|---|---:|---|")
        tables = []
        for label, agg in summaries.items():
            s = agg.get(subset)
            if s is None or s["n"] == 0:
                lines.append(f"| {label} | - | 0 | n/a | n/a | n/a | n/a |")
                continue
            for cond in CONDITIONS:
                c = s["conditions"].get(cond)
                if c is None:
                    continue
                lines.append(f"| {label} | {cond} | {s['n']} | {s['clean_accuracy']:.3f} | "
                             f"{_fmt(c['mean_agreement'])} | {c['k90']}/{s['num_frames']} | "
                             f"{_fmt(c['min_sufficient_k'], 1)} |")
            tables.append((label, s))
        lines.append("")
        for label, s in tables:
            header = " | ".join(f"{k}" for k in s["budgets"])
            lines.append(f"Agreement with the full-clip prediction per frames kept (of {s['num_frames']}):")
            lines.append("")
            lines.append(f"| {label} | {header} |")
            lines.append("|---|" + "---:|" * len(s["budgets"]))
            for cond in CONDITIONS:
                c = s["conditions"].get(cond)
                if c is not None:
                    lines.append(f"| {cond} | " + " | ".join(f"{a['mean']:.2f}" for a in c["agreement"]) + " |")
            lines.append("")
            if subset == "all":
                lines.append("Accuracy per frames kept:")
                lines.append("")
                lines.append(f"| {label} | {header} |")
                lines.append("|---|" + "---:|" * len(s["budgets"]))
                for cond in CONDITIONS:
                    c = s["conditions"].get(cond)
                    if c is not None:
                        lines.append(f"| {cond} | " + " | ".join(f"{a['mean']:.2f}" for a in c["accuracy"]) + " |")
                lines.append("")
    return "\n".join(lines) + "\n"


def write_curves_csv(path, summaries):
    with open(path, "w") as f:
        f.write("label,subset,condition,k,num_frames,agreement,agreement_ci_low,agreement_ci_high,"
                "accuracy,accuracy_ci_low,accuracy_ci_high,p_pred_clean\n")
        for label, agg in summaries.items():
            for subset, s in agg.items():
                if s["n"] == 0:
                    continue
                for cond, c in s["conditions"].items():
                    for b, k in enumerate(s["budgets"]):
                        a, acc = c["agreement"][b], c["accuracy"][b]
                        f.write(f"{label},{subset},{cond},{k},{s['num_frames']},{a['mean']:.4f},{a['ci_low']:.4f},"
                                f"{a['ci_high']:.4f},{acc['mean']:.4f},{acc['ci_low']:.4f},{acc['ci_high']:.4f},"
                                f"{c['p_pred_clean'][b]:.4f}\n")


def plot_curves(path, summaries, subset=None):
    """Agreement vs frames kept, one panel per model, one line per condition with its CI band."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping curves.png")
        return
    panels = []
    for label, agg in summaries.items():
        chosen = subset
        if chosen is None:
            chosen = next((name for name in SUBSET_ORDER if agg.get(name, {}).get("n", 0) > 0), None)
        if chosen is not None and agg.get(chosen, {}).get("n", 0) > 0:
            panels.append((label, chosen, agg[chosen]))
    if not panels:
        return
    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False, sharey=True)
    handles = {}
    for ax, (label, chosen, s) in zip(axes.reshape(-1), panels):
        ks = np.array(s["budgets"])
        for cond in CONDITIONS:
            c = s["conditions"].get(cond)
            if c is None:
                continue
            color, style, marker = PLOT_STYLE[cond]
            mean = np.array([a["mean"] for a in c["agreement"]])
            low = np.array([a["ci_low"] for a in c["agreement"]])
            high = np.array([a["ci_high"] for a in c["agreement"]])
            ax.fill_between(ks, low, high, color=color, alpha=0.12, linewidth=0)
            (line,) = ax.plot(ks, mean, style, color=color, marker=marker, markersize=4, linewidth=1.6, label=cond)
            handles[cond] = line
        ax.axhline(0.9, color="#888888", linewidth=0.8, linestyle=(0, (2, 3)))
        ax.set_title(f"{label} ({chosen}, n={s['n']})", fontsize=10)
        ax.set_xlabel(f"frames kept (of {s['num_frames']})")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.reshape(-1)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("agreement with full-clip prediction")
    fig.legend([handles[c] for c in CONDITIONS if c in handles], [c for c in CONDITIONS if c in handles],
               loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_clip_results(out_dir):
    clips_dir = os.path.join(out_dir, "clips")
    rows = []
    if os.path.isdir(clips_dir):
        for name in sorted(os.listdir(clips_dir)):
            if name.endswith(".json"):
                with open(os.path.join(clips_dir, name)) as f:
                    rows.append(json.load(f))
    return rows


def write_summary(out_dir, label, clips, settings):
    summary = aggregate(clips)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"label": label, "settings": settings, "summary": summary}, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown({label: summary}, settings))
    write_curves_csv(os.path.join(out_dir, "curves.csv"), {label: summary})
    plot_curves(os.path.join(out_dir, "curves.png"), {label: summary})
    return summary


# ============================================================================
# Driver
# ============================================================================

def forwards_per_clip(budgets, random_seeds, transfer):
    return 1 + len(budgets) * (3 + random_seeds + int(transfer))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, help="Model that is re-run on the selected frames.")
    parser.add_argument("--config_path", help="Same config as the attribution scripts (test split + keywords).")
    parser.add_argument("--saliency_dir", help="Saved <clip>.npz / <clip>.json of this model (frame relevance).")
    parser.add_argument("--out_dir", help="Where per-clip results and the summary are written.")
    parser.add_argument("--label", default=None, help="Row label in the summary (default: --model).")
    parser.add_argument("--map_key", default="saliency_maps",
                        help="npz array whose per-frame sum ranks the frames (default saliency_maps).")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS),
                        help="Frame budgets k to test (budgets >= T are dropped; T itself is the clean run).")
    parser.add_argument("--mode", choices=("mask", "drop"), default="mask",
                        help="mask: unselected frames mean-filled; drop: model sees only the k frames.")
    parser.add_argument("--fill", choices=("mean", "zero"), default="mean")
    parser.add_argument("--random_seeds", type=int, default=1, help="Random k-frame draws per clip and budget.")
    parser.add_argument("--logit_scale", type=float, default=None,
                        help="Override the softmax temperature applied to the logits.")
    parser.add_argument("--rank_saliency_dir", default=None,
                        help="Another model's saliency dir: adds the 'transfer' condition ranked by its maps.")
    parser.add_argument("--rank_label", default=None, help="Name of the ranking model (for the report).")
    parser.add_argument("--max_clips", type=int, default=None, help="Debug: stop after this many clips.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute clips that already have results.")
    parser.add_argument("--summarize_only", action="store_true",
                        help="Rebuild summary.* from <out_dir>/clips without loading a model.")
    parser.add_argument("--summarize_runs", nargs="*", default=None, metavar="LABEL=OUT_DIR",
                        help="Combine several finished runs into one table and figure written to --out_dir.")
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
            with open(os.path.join(run_dir, "summary.json")) as f:
                saved = json.load(f)
            summaries[label] = saved["summary"]
            settings = saved.get("settings", settings)
        with open(os.path.join(args.out_dir, "summary_all.md"), "w") as f:
            f.write(summary_markdown(summaries, settings))
        with open(os.path.join(args.out_dir, "summary_all.json"), "w") as f:
            json.dump({"settings": settings, "summary": summaries}, f, indent=2)
        write_curves_csv(os.path.join(args.out_dir, "curves_all.csv"), summaries)
        plot_curves(os.path.join(args.out_dir, "curves_all.png"), summaries)
        print(f"Wrote {os.path.join(args.out_dir, 'summary_all.md')} and curves_all.png")
        return

    label = args.label or args.model
    settings = {"budgets": list(args.budgets), "mode": args.mode, "fill": args.fill, "map_key": args.map_key,
                "random_seeds": args.random_seeds, "rank_from": args.rank_label or args.rank_saliency_dir}

    if args.summarize_only:
        clips = load_clip_results(args.out_dir)
        if not clips:
            raise SystemExit(f"no per-clip results under {args.out_dir}/clips")
        settings = clips[0].get("settings", settings)
        summary = write_summary(args.out_dir, label, clips, settings)
        print(summary_markdown({label: summary}, settings))
        return

    for required in ("model", "config_path", "saliency_dir"):
        if getattr(args, required) is None:
            raise SystemExit(f"--{required} is required")
    if args.mode == "drop" and args.model not in FRAME_INDEPENDENT_MODELS:
        print(f"WARNING: --mode drop with {args.model}: the model has a fixed-length temporal embedding and "
              f"will most likely fail on fewer frames; use --mode mask.")

    from dataset.video_utils_siglip import read_frames_decord
    from grounding_metrics import load_saliency_dir  # needs cv2; only the model run does

    test_config = load_config(args.config_path)["dataset"]["test"]
    keywords = test_config["keywords"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = {clip_id: (frame_relevance(maps), meta)
             for clip_id, maps, meta in load_saliency_dir(args.saliency_dir, args.map_key)}
    print(f"[{label}] {len(saved)} saved maps in {args.saliency_dir} (key {args.map_key!r})")
    ranking = None
    if args.rank_saliency_dir:
        ranking = {clip_id: (frame_relevance(maps), meta)
                   for clip_id, maps, meta in load_saliency_dir(args.rank_saliency_dir, args.map_key)}
        print(f"[{label}] transfer condition: {len(ranking)} maps from {args.rank_saliency_dir}")

    predictor = build_predictor(args, keywords, device, load_config(args.config_path))
    logit_scale = args.logit_scale if args.logit_scale is not None else predictor.logit_scale
    settings.update({"logit_scale": logit_scale, "grid": predictor.grid, "model": args.model})
    print(f"[{label}] mode={args.mode}, softmax temperature {logit_scale:.3f}")

    clips_dir = os.path.join(args.out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    started = time.time()
    n_done = n_mismatch = 0
    for index, (clip_id, video_path, caption) in enumerate(iter_clips(test_config)):
        if args.max_clips is not None and index >= args.max_clips:
            break
        stem = sanitize(clip_id)
        result_path = os.path.join(clips_dir, stem + ".json")
        if not args.overwrite and os.path.isfile(result_path):
            continue
        if clip_id not in saved:
            print(f"  [{index + 1}] {clip_id}: no saved map, skipped")
            continue
        relevance, meta = saved[clip_id]
        other = other_meta = None
        if ranking is not None:
            if clip_id not in ranking:
                print(f"  [{index + 1}] {clip_id}: no map in --rank_saliency_dir, skipped")
                continue
            other, other_meta = ranking[clip_id]
        target = int(meta.get("target_index", keywords.index(meta.get("ground_truth_text", caption))))

        frames, frame_indices, _ = read_frames_decord(video_path, args.num_frames, test_config.get("sample", "middle"))
        if frames.shape[0] != len(relevance):
            raise ValueError(f"{clip_id}: {frames.shape[0]} frames but the map has {len(relevance)}")
        x = predictor.preprocess(frames)
        if x.shape[0] != len(relevance) or x.shape[-1] % predictor.grid:
            raise ValueError(f"{clip_id}: preprocessed input {tuple(x.shape)} does not match grid {predictor.grid}")
        budgets = sorted({int(k) for k in args.budgets if 0 < k < x.shape[0]})
        if index == 0 or n_done == 0:
            print(f"[{label}] budgets {budgets} of {x.shape[0]} frames, "
                  f"{forwards_per_clip(budgets, args.random_seeds, other is not None)} forwards per clip")

        seeds = [1000 * index + s for s in range(args.random_seeds)]
        try:
            result = score_clip(predictor, x, relevance, target, budgets, args.mode, args.fill, seeds, logit_scale,
                                other)
        except RuntimeError as err:
            if args.mode == "drop":
                raise SystemExit(f"{args.model} failed on a reduced number of frames ({err}); "
                                 f"use --mode mask for this model.") from err
            raise
        pred_text = keywords[result["pred_clean"]]
        meta_pred = meta.get("prediction_text")
        mismatch = meta_pred is not None and str(meta_pred).strip().lower() != pred_text.lower()
        n_mismatch += mismatch
        result.update({
            "clip_id": clip_id,
            "target_index": target,
            "target_text": keywords[target],
            "pred_clean_text": pred_text,
            "saved_prediction_text": meta_pred,
            "clean_prediction_matches_saved": not mismatch,
            "frame_indices": [int(i) for i in frame_indices],
            "settings": settings,
        })
        if other_meta is not None:
            gt = str(other_meta.get("ground_truth_text", caption)).strip().lower()
            result["transfer_saved_correct"] = str(other_meta.get("prediction_text", "")).strip().lower() == gt
        with open(result_path, "w") as f:
            json.dump(result, f)
        n_done += 1
        mins = {c: minimal_sufficient_k(result, c) for c in CONDITIONS if c in result and result[c]}
        print(f"  [{index + 1}] {clip_id}: target={keywords[target]!r} pred={pred_text!r}"
              f"{' (differs from saved run!)' if mismatch else ''} p={result['p_clean_target']:.3f} "
              f"min-k " + " ".join(f"{c}={v:g}" for c, v in mins.items()) + f" ({time.time() - started:.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_mismatch:
        print(f"WARNING: {n_mismatch} clips where the clean prediction differs from the saved run's "
              f"prediction_text; check that frames, checkpoint and preprocessing match.")
    clips = load_clip_results(args.out_dir)
    summary = write_summary(args.out_dir, label, clips, settings)
    print(f"[{label}] scored {n_done} new clips, {len(clips)} total -> {args.out_dir}/summary.md")
    print(summary_markdown({label: summary}, settings))


if __name__ == "__main__":
    main()
