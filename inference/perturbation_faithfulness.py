"""Perturbation-based faithfulness of the saved saliency maps.

Grounding metrics (Table 1) ask whether a map lands on the annotated cue.
This script asks the complementary question: does the model actually *use*
what the map highlights? It never recomputes attribution; it reads the
`<clip>.npz` / `<clip>.json` pairs a Table 1 run wrote with
`--saliency_save_dir`, re-loads the same 30 frames, masks tokens according to
the saved map and re-runs the classifier forward pass.

Protocol (per clip)
-------------------
The (T, h, w) saliency volume is ranked over all T*h*w tokens (``--ranking
clip``) or within each frame (``--ranking frame``) and split into K
equally sized groups G_1 (most salient) ... G_K (``--groups``, default 10,
i.e. 0-10%, 10-20%, ...). Each token is a model patch (16x16 px for SigLIP,
18x18 for VideoPrism, ...), so masking is done on whole patches of the
model's own square input, replaced by the per-clip channel mean
(``--fill mean``, Wu et al. 2024) or zero. Three families of perturbed
inputs are scored with the explained class' softmax probability:

* independent   one group at a time               -> SaCo (Wu et al., CVPR 2024)
* MoRF          cumulative, most relevant first   -> deletion curve, AOPC,
                                                     flip rate, first-flip fraction
* LeRF          cumulative, least relevant first  -> insertion-style control

Every clip is also scored with a uniformly random saliency volume under the
same protocol (``--random_seeds``). A faithful map must beat this control:
random deletion measures what masking alone does to an off-manifold input,
and SaCo of a random map is 0 in expectation.

SaCo (Algorithm 1 of arXiv:2404.01415): with s(G_i) the summed salience of a
group and d_i the probability drop when only G_i is masked,

    F = sum_{i<j} sign(d_i >= d_j) * (s_i - s_j)  /  sum_{i<j} |s_i - s_j|

so F in [-1, 1]; 1 means the drops are perfectly ordered like the salience.

Readout
-------
p = softmax(logit_scale * logits). The supervised heads (MatchVision,
SoccerMaster) use scale 1. The zero-shot models produce cosine similarities,
so their own learned temperature is applied (SigLIP / CLIP: exp(logit_scale)
from the checkpoint; PE-AV: text_video_logit_scale; VideoPrism: 1, the
temperature its forward uses). ``--logit_scale`` overrides. The flip metrics
only use argmax and do not depend on the scale.

Outputs
-------
    <out_dir>/clips/<stem>.json   per-clip curves and scores (resumable)
    <out_dir>/summary.json        aggregates for all clips and for the
                                  correctly classified subset, with bootstrap CIs
    <out_dir>/summary.md          the same as tables
    <out_dir>/curves.csv          mean deletion curves per masked fraction

``--summarize_runs Label=<out_dir> ...`` builds one table across several
finished runs without a model.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODELS = ("siglip", "matchvision", "soccermaster", "clip", "peav", "videoprism")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ============================================================================
# Grouping and masking (model-agnostic)
# ============================================================================

def rank_groups(saliency, num_groups, ranking="clip"):
    """Assign every token of a (T, h, w) volume to a salience group.

    Returns an int array of the same shape with values in [0, num_groups):
    0 is the most salient group. ``clip`` ranks all T*h*w tokens together;
    ``frame`` ranks within each frame so every frame contributes equally to
    every group. Ties keep index order (stable sort).
    """
    saliency = np.asarray(saliency, dtype=np.float64)
    if saliency.ndim != 3:
        raise ValueError(f"saliency must be (T, h, w), got {saliency.shape}")
    groups = np.empty(saliency.shape, dtype=np.int64)
    if ranking == "clip":
        order = np.argsort(-saliency.reshape(-1), kind="stable")
        flat = groups.reshape(-1)
        for g, idx in enumerate(np.array_split(order, num_groups)):
            flat[idx] = g
    elif ranking == "frame":
        for t in range(saliency.shape[0]):
            order = np.argsort(-saliency[t].reshape(-1), kind="stable")
            flat = groups[t].reshape(-1)
            for g, idx in enumerate(np.array_split(order, num_groups)):
                flat[idx] = g
    else:
        raise ValueError(f"unknown ranking {ranking!r}")
    return groups


def group_salience(saliency, groups, num_groups):
    """s(G_i): summed salience per group (Eq. 1)."""
    saliency = np.asarray(saliency, dtype=np.float64).reshape(-1)
    flat = groups.reshape(-1)
    return np.array([saliency[flat == g].sum() for g in range(num_groups)])


def mask_tokens(x, token_mask, fill="mean"):
    """Replace masked patches of ``x`` (T, 3, S, S) by a constant.

    ``token_mask`` is a bool (T, h, w) array on the model's token grid; each
    token covers an (S/h) x (S/w) patch. ``mean`` uses the per-clip channel
    mean of ``x`` (the per-sample mean of Wu et al.), ``zero`` uses 0 in the
    model's normalized input space.
    """
    num_frames, channels, height, width = x.shape
    grid_t, grid_h, grid_w = token_mask.shape
    if grid_t != num_frames or height % grid_h or width % grid_w:
        raise ValueError(f"token grid {token_mask.shape} does not tile input {tuple(x.shape)}")
    mask = torch.as_tensor(np.ascontiguousarray(token_mask), device=x.device)
    mask = mask.repeat_interleave(height // grid_h, dim=1).repeat_interleave(width // grid_w, dim=2)
    if fill == "mean":
        value = x.mean(dim=(0, 2, 3))
    elif fill == "zero":
        value = torch.zeros(channels, dtype=x.dtype, device=x.device)
    else:
        raise ValueError(f"unknown fill {fill!r}")
    return torch.where(mask[:, None], value[None, :, None, None].to(x.dtype), x)


# ============================================================================
# Scores
# ============================================================================

def saco(drops, salience):
    """Salience-guided Faithfulness Coefficient (Wu et al. 2024, Alg. 1)."""
    drops = np.asarray(drops, dtype=np.float64)
    salience = np.asarray(salience, dtype=np.float64)
    total = coef = 0.0
    for i in range(len(drops)):
        for j in range(i + 1, len(drops)):
            weight = salience[i] - salience[j]
            coef += weight if drops[i] >= drops[j] else -weight
            total += abs(weight)
    return float(coef / total) if total > 0 else 0.0


def curve_scores(p_clean, p_steps, pred_clean, pred_steps, num_groups):
    """Scores of a cumulative deletion curve.

    ``p_steps[i]`` is the target probability after masking groups 0..i
    (fraction (i+1)/K); the last step is the fully masked input.
    """
    p_steps = np.asarray(p_steps, dtype=np.float64)
    fractions = np.concatenate([[0.0], np.arange(1, num_groups + 1) / num_groups])
    curve = np.concatenate([[p_clean], p_steps])
    flipped = [int(p) != int(pred_clean) for p in pred_steps]
    first_flip = next((fractions[i + 1] for i, f in enumerate(flipped) if f), 1.0)
    return {
        "p_target": p_steps.tolist(),
        "pred": [int(p) for p in pred_steps],
        "flipped": flipped,
        # Average drop over the partial steps (Samek et al. 2017); the fully
        # masked step is the same input for every method and is left out.
        "aopc": float(np.mean(p_clean - p_steps[:-1])) if num_groups > 1 else float("nan"),
        "auc": float(_trapezoid(curve, fractions)),
        "first_flip_fraction": float(first_flip),
    }


def score_clip(predictor, x, saliency, target, num_groups, ranking, fill, random_seeds, logit_scale):
    """Run the full protocol for one clip; returns a JSON-serializable dict."""
    def probs(inp):
        logits = predictor.forward(inp).detach().float().cpu()
        return torch.softmax(logit_scale * logits, dim=-1).numpy(), int(logits.argmax())

    with torch.no_grad():
        p_clean, pred_clean = probs(x)
        full = mask_tokens(x, np.ones(saliency.shape, dtype=bool), fill)
        p_full, pred_full = probs(full)

        def run(volume):
            groups = rank_groups(volume, num_groups, ranking)
            salience = group_salience(volume, groups, num_groups)
            independent = []
            for g in range(num_groups):
                p, pred = probs(mask_tokens(x, groups == g, fill))
                independent.append((float(p[target]), pred))
            morf, lerf = [], []
            for i in range(num_groups - 1):
                p, pred = probs(mask_tokens(x, groups <= i, fill))
                morf.append((float(p[target]), pred))
                p, pred = probs(mask_tokens(x, groups >= num_groups - 1 - i, fill))
                lerf.append((float(p[target]), pred))
            morf.append((float(p_full[target]), pred_full))
            lerf.append((float(p_full[target]), pred_full))
            drops = [p_clean[target] - p for p, _ in independent]
            return {
                "group_salience": salience.tolist(),
                "independent": {
                    "p_target": [p for p, _ in independent],
                    "drop": [float(d) for d in drops],
                    "pred": [pred for _, pred in independent],
                    "flipped": [pred != pred_clean for _, pred in independent],
                },
                "saco": saco(drops, salience),
                "morf": curve_scores(p_clean[target], [p for p, _ in morf], pred_clean,
                                     [pred for _, pred in morf], num_groups),
                "lerf": curve_scores(p_clean[target], [p for p, _ in lerf], pred_clean,
                                     [pred for _, pred in lerf], num_groups),
            }

        result = {"saliency": run(saliency), "random": []}
        for seed in random_seeds:
            volume = np.random.default_rng(seed).random(saliency.shape)
            entry = run(volume)
            entry["seed"] = int(seed)
            result["random"].append(entry)

    result.update({
        "p_clean": p_clean.tolist(),
        "p_clean_target": float(p_clean[target]),
        "pred_clean": pred_clean,
        "p_full_mask_target": float(p_full[target]),
        "pred_full_mask": pred_full,
    })
    return result


# ============================================================================
# Aggregation
# ============================================================================

def bootstrap_mean_ci(values, draws=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(len(values)),
    }


def _random_mean(clip, key_path):
    """Mean over the random draws of a clip of a nested scalar."""
    vals = []
    for entry in clip["random"]:
        node = entry
        for key in key_path:
            node = node[key]
        vals.append(node)
    return float(np.mean(vals)) if vals else float("nan")


def _random_curve(clip, family, key):
    return np.mean([entry[family][key] for entry in clip["random"]], axis=0)


def aggregate(clips):
    """Aggregate per-clip results; returns {subset: {...}} for all / correct clips."""
    out = {}
    subsets = {"all": clips, "correct": [c for c in clips if c["pred_clean"] == c["target_index"]]}
    for name, rows in subsets.items():
        if not rows:
            out[name] = {"n": 0}
            continue
        num_groups = len(rows[0]["saliency"]["group_salience"])
        fractions = (np.arange(1, num_groups + 1) / num_groups).tolist()
        sal = [r["saliency"] for r in rows]
        summary = {
            "n": len(rows),
            "fractions": fractions,
            "saco": bootstrap_mean_ci([s["saco"] for s in sal]),
            "saco_random": bootstrap_mean_ci([_random_mean(r, ("saco",)) for r in rows]),
            "saco_minus_random": bootstrap_mean_ci([s["saco"] - _random_mean(r, ("saco",)) for s, r in zip(sal, rows)]),
            "frac_clips_saco_above_random": float(np.mean(
                [s["saco"] > _random_mean(r, ("saco",)) for s, r in zip(sal, rows)])),
            "aopc_morf": bootstrap_mean_ci([s["morf"]["aopc"] for s in sal]),
            "aopc_morf_random": bootstrap_mean_ci([_random_mean(r, ("morf", "aopc")) for r in rows]),
            "aopc_lerf": bootstrap_mean_ci([s["lerf"]["aopc"] for s in sal]),
            "auc_morf": bootstrap_mean_ci([s["morf"]["auc"] for s in sal]),
            "auc_morf_random": bootstrap_mean_ci([_random_mean(r, ("morf", "auc")) for r in rows]),
            "auc_lerf": bootstrap_mean_ci([s["lerf"]["auc"] for s in sal]),
            "auc_gap_lerf_minus_morf": bootstrap_mean_ci([s["lerf"]["auc"] - s["morf"]["auc"] for s in sal]),
            "first_flip_fraction_morf": bootstrap_mean_ci([s["morf"]["first_flip_fraction"] for s in sal]),
            "first_flip_fraction_random": bootstrap_mean_ci(
                [_random_mean(r, ("morf", "first_flip_fraction")) for r in rows]),
            "first_flip_fraction_lerf": bootstrap_mean_ci([s["lerf"]["first_flip_fraction"] for s in sal]),
            "p_clean_target": bootstrap_mean_ci([r["p_clean_target"] for r in rows]),
            "p_full_mask_target": bootstrap_mean_ci([r["p_full_mask_target"] for r in rows]),
            "curves": {
                "morf_p_target": np.mean([s["morf"]["p_target"] for s in sal], axis=0).tolist(),
                "lerf_p_target": np.mean([s["lerf"]["p_target"] for s in sal], axis=0).tolist(),
                "random_p_target": np.mean([_random_curve(r, "morf", "p_target") for r in rows], axis=0).tolist(),
                "morf_flip_rate": np.mean([s["morf"]["flipped"] for s in sal], axis=0).tolist(),
                "lerf_flip_rate": np.mean([s["lerf"]["flipped"] for s in sal], axis=0).tolist(),
                "random_flip_rate": np.mean([_random_curve(r, "morf", "flipped") for r in rows], axis=0).tolist(),
                "independent_drop": np.mean([s["independent"]["drop"] for s in sal], axis=0).tolist(),
                "independent_flip_rate": np.mean([s["independent"]["flipped"] for s in sal], axis=0).tolist(),
                "random_independent_drop": np.mean(
                    [_random_curve(r, "independent", "drop") for r in rows], axis=0).tolist(),
            },
        }
        out[name] = summary
    return out


def _fmt(stat, digits=3):
    if not stat or stat.get("n", 0) == 0 or np.isnan(stat["mean"]):
        return "n/a"
    return f"{stat['mean']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def summary_markdown(summaries, settings):
    """``summaries`` maps a run label to the dict from ``aggregate``."""
    lines = ["# Perturbation faithfulness", ""]
    lines.append("Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    lines.append("")
    lines.append("Values are means over clips with 95% bootstrap CIs. SaCo in [-1, 1] (random map: 0). "
                 "AOPC = mean target-probability drop over the partial MoRF steps (higher = more faithful; "
                 "compare with the random row, which measures the masking artifact alone). "
                 "AUC is the area under the target-probability deletion curve (lower MoRF, higher LeRF = better). "
                 "First flip = smallest masked fraction at which the top-1 prediction changes "
                 "(1.0 = never before full masking).")
    for subset in ("all", "correct"):
        lines += ["", f"## {'All clips' if subset == 'all' else 'Correctly classified clips'}", ""]
        lines.append("| Model | n | SaCo | SaCo (random) | SaCo - random | clips > random | AOPC MoRF | AOPC random | "
                     "AOPC LeRF | AUC MoRF | AUC random | AUC LeRF | first flip MoRF | first flip random | "
                     "first flip LeRF |")
        lines.append("|---|---:|---|---|---|---:|---|---|---|---|---|---|---|---|---|")
        for label, summary in summaries.items():
            s = summary.get(subset, {})
            if not s or s.get("n", 0) == 0:
                lines.append(f"| {label} | 0 | " + " | ".join(["n/a"] * 13) + " |")
                continue
            lines.append(
                f"| {label} | {s['n']} | {_fmt(s['saco'])} | {_fmt(s['saco_random'])} | {_fmt(s['saco_minus_random'])} | "
                f"{100 * s['frac_clips_saco_above_random']:.0f}% | {_fmt(s['aopc_morf'])} | {_fmt(s['aopc_morf_random'])} | "
                f"{_fmt(s['aopc_lerf'])} | {_fmt(s['auc_morf'])} | {_fmt(s['auc_morf_random'])} | {_fmt(s['auc_lerf'])} | "
                f"{_fmt(s['first_flip_fraction_morf'], 2)} | {_fmt(s['first_flip_fraction_random'], 2)} | "
                f"{_fmt(s['first_flip_fraction_lerf'], 2)} |")
        lines += ["", "Per masked fraction (cumulative MoRF vs. a random map; LeRF for reference):", ""]
        for label, summary in summaries.items():
            s = summary.get(subset, {})
            if not s or s.get("n", 0) == 0:
                continue
            fr = s["fractions"]
            lines.append(f"| {label} | " + " | ".join(f"{f:.0%}" for f in fr) + " |")
            lines.append("|---|" + "---:|" * len(fr))
            lines.append("| MoRF flip rate | " + " | ".join(f"{v:.2f}" for v in s["curves"]["morf_flip_rate"]) + " |")
            lines.append("| random flip rate | " + " | ".join(f"{v:.2f}" for v in s["curves"]["random_flip_rate"]) + " |")
            lines.append("| MoRF p(target) | " + " | ".join(f"{v:.3f}" for v in s["curves"]["morf_p_target"]) + " |")
            lines.append("| random p(target) | " + " | ".join(f"{v:.3f}" for v in s["curves"]["random_p_target"]) + " |")
            lines.append("| LeRF p(target) | " + " | ".join(f"{v:.3f}" for v in s["curves"]["lerf_p_target"]) + " |")
            lines.append("| single-group drop | " + " | ".join(f"{v:.3f}" for v in s["curves"]["independent_drop"]) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_curves_csv(path, summaries):
    with open(path, "w") as f:
        f.write("run,subset,fraction,morf_p_target,lerf_p_target,random_p_target,morf_flip_rate,lerf_flip_rate,"
                "random_flip_rate,independent_drop,random_independent_drop,independent_flip_rate\n")
        for label, summary in summaries.items():
            for subset, s in summary.items():
                if not s or s.get("n", 0) == 0:
                    continue
                c = s["curves"]
                for i, fr in enumerate(s["fractions"]):
                    f.write(f"{label},{subset},{fr:.4f},{c['morf_p_target'][i]:.6f},{c['lerf_p_target'][i]:.6f},"
                            f"{c['random_p_target'][i]:.6f},{c['morf_flip_rate'][i]:.4f},{c['lerf_flip_rate'][i]:.4f},"
                            f"{c['random_flip_rate'][i]:.4f},{c['independent_drop'][i]:.6f},"
                            f"{c['random_independent_drop'][i]:.6f},{c['independent_flip_rate'][i]:.4f}\n")


def load_clip_results(out_dir):
    clips_dir = os.path.join(out_dir, "clips")
    rows = []
    if not os.path.isdir(clips_dir):
        return rows
    for name in sorted(os.listdir(clips_dir)):
        if name.endswith(".json"):
            with open(os.path.join(clips_dir, name)) as f:
                rows.append(json.load(f))
    return rows


def write_summary(out_dir, label, clips, settings):
    summaries = {label: aggregate(clips)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"label": label, "settings": settings, "summary": summaries[label]}, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown(summaries, settings))
    write_curves_csv(os.path.join(out_dir, "curves.csv"), summaries)
    return summaries[label]


# ============================================================================
# Model predictors: preprocess(uint8 frames) -> (T, 3, S, S); forward(x) -> (C,) logits
# ============================================================================

def read_checkpoint_scalar(name, key, revision=None):
    """Read one scalar tensor (e.g. ``logit_scale``) from a Hub safetensors checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(name, "model.safetensors", revision=revision)
    with safe_open(path, framework="pt") as f:
        if key not in f.keys():
            raise KeyError(f"{name} has no tensor {key!r}")
        return f.get_tensor(key).float().reshape(-1)[0].item()


def _siglip_processor_frames(processor, frames_uint8):
    """Exactly what `VideoCaptionDataset.__getitem__` does per frame."""
    return torch.cat([processor(images=frame, return_tensors="pt")["pixel_values"] for frame in frames_uint8], dim=0)


class SiglipPredictor:
    """`SigLIP_Classifier` (zero-shot SigLIP, mean of frame embeddings)."""

    def __init__(self, args, keywords, device):
        from model.SigLIP_classifier import SigLIP_Classifier
        from dataset.video_utils_siglip import set_transform

        self.device = device
        self.model = SigLIP_Classifier(keywords=keywords, feature_dim=768, model_name=args.model_name,
                                       temporal_aggregation=args.siglip_temporal_aggregation).eval().to(device)
        self.processor = set_transform()
        cfg = self.model.siglip_model.config
        self.grid = cfg.image_size // cfg.patch_size
        # SigLIP's learned temperature; the bias cancels in a softmax over classes.
        self.logit_scale = float(np.exp(read_checkpoint_scalar(args.model_name, "logit_scale")))

    def preprocess(self, frames_uint8):
        return _siglip_processor_frames(self.processor, frames_uint8).to(self.device)

    def forward(self, x):
        return self.model(x[None])[0]  # accepts (B, T, C, H, W)


class MatchVisionPredictor:
    def __init__(self, args, keywords, device, config):
        from inference_utils import load_classifier
        from dataset.video_utils_siglip import set_transform

        settings = config["training_settings"]
        self.device = device
        device_ids = [device.index if device.index is not None else 0]
        self.model = load_classifier(config["dataset"]["test"], settings["classifier_transformer_type"],
                                     settings["encoder_type"], settings["use_transformer"],
                                     args.checkpoint_path, [device], device_ids).module.eval()
        self.processor = set_transform()
        self.grid = 224 // 16
        self.logit_scale = 1.0

    def preprocess(self, frames_uint8):
        return _siglip_processor_frames(self.processor, frames_uint8).to(self.device)

    def forward(self, x):
        # (T, 3, H, W) -> (1, 3, T, H, W); the classifier prints a shape per call.
        with contextlib.redirect_stdout(io.StringIO()):
            return self.model(x.permute(1, 0, 2, 3)[None])[0]


class SoccerMasterPredictor:
    def __init__(self, args, keywords, device):
        from chefer_soccermaster import load_soccermaster_model
        from dataset.video_utils_siglip import set_transform

        self.device = device
        self.model = load_soccermaster_model(args.checkpoint_dir, device, siglip2_path=args.siglip2_path,
                                             num_frames=args.num_frames)
        self.processor = set_transform()
        self.input_size = args.input_size
        self.grid = args.input_size // 16
        self.logit_scale = 1.0

    def preprocess(self, frames_uint8):
        # The inference scripts normalize at 224 with the SigLIP-base processor,
        # then resize to 512; keep that order so the clean logits match.
        x = _siglip_processor_frames(self.processor, frames_uint8).to(self.device)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x.half()

    def forward(self, x):
        backbone_out = self.model.backbone(x[None])
        return self.model.multi_task_head["CaptionClassification"](backbone_out, metas=None)["logits"][0]


class BaselinePredictor:
    """CLIP / PE-AV / VideoPrism through the adapters of `chefer_baselines.py`."""

    def __init__(self, args, keywords, device):
        from attention_recorder import AttentionRecorder
        from chefer_baselines import ADAPTERS, CHECKPOINTS

        name, revision = CHECKPOINTS[args.model]
        if args.checkpoint:
            name, revision = args.checkpoint, args.revision
        recorder = AttentionRecorder()
        recorder.enabled = False
        self.adapter = ADAPTERS[args.model](keywords, device, recorder, name, revision)
        self.grid = self.adapter.grid
        self.logit_scale = float(self.adapter.logit_scale)

    def preprocess(self, frames_uint8):
        return self.adapter.preprocess(frames_uint8)

    def forward(self, x):
        return self.adapter.forward_logits(x)


def build_predictor(args, keywords, device, config):
    if args.model == "siglip":
        return SiglipPredictor(args, keywords, device)
    if args.model == "matchvision":
        if not args.checkpoint_path:
            raise SystemExit("--checkpoint_path is required for matchvision")
        return MatchVisionPredictor(args, keywords, device, config)
    if args.model == "soccermaster":
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint_dir is required for soccermaster")
        return SoccerMasterPredictor(args, keywords, device)
    return BaselinePredictor(args, keywords, device)


# ============================================================================
# Driver
# ============================================================================

def load_config(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def sanitize(clip_id):
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(clip_id))


def iter_clips(test_config):
    for json_path, video_base in zip(test_config["json"], test_config["video_base"]):
        with open(json_path) as f:
            items = json.load(f)
        for item in items:
            variant = item.get("variant")
            clip_id = item["video"] + (f"#{variant}" if variant else "")
            yield clip_id, os.path.join(video_base, item["video"]), item["caption"]


def forwards_per_clip(num_groups, random_seeds):
    """clean + full mask + K independent + 2(K-1) cumulative, then per random
    draw K independent + 2(K-1) cumulative (the full mask is shared)."""
    return 2 + (3 * num_groups - 2) * (1 + random_seeds)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, help="Model whose saved maps are tested.")
    parser.add_argument("--config_path", help="Same config as the attribution scripts (test split + keywords).")
    parser.add_argument("--saliency_dir", help="Directory of <clip>.npz / <clip>.json pairs from a finished run.")
    parser.add_argument("--out_dir", help="Where per-clip results and the summary are written.")
    parser.add_argument("--label", default=None, help="Row label in the summary (default: --model).")
    parser.add_argument("--map_key", default="saliency_maps",
                        help="npz array to rank: saliency_maps (Table 1 map) or chefer_spatial_maps.")
    parser.add_argument("--groups", type=int, default=10,
                        help="K equally sized salience groups (default 10 = 10%% bins).")
    parser.add_argument("--ranking", choices=("clip", "frame"), default="clip",
                        help="Rank tokens over the whole clip volume or within each frame.")
    parser.add_argument("--fill", choices=("mean", "zero"), default="mean",
                        help="Replacement value for masked patches.")
    parser.add_argument("--random_seeds", type=int, default=1, help="Random-map controls per clip (default 1).")
    parser.add_argument("--logit_scale", type=float, default=None,
                        help="Override the softmax temperature applied to the logits.")
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--max_clips", type=int, default=None, help="Debug: stop after this many clips.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute clips that already have results.")
    parser.add_argument("--summarize_only", action="store_true",
                        help="Rebuild summary.* from <out_dir>/clips without loading a model.")
    parser.add_argument("--summarize_runs", nargs="*", default=None, metavar="LABEL=OUT_DIR",
                        help="Combine several finished runs into one table written to --out_dir.")
    # SigLIP
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224")
    parser.add_argument("--siglip_temporal_aggregation", default="mean", choices=("mean", "max"))
    # MatchVision
    parser.add_argument("--checkpoint_path", default=None)
    # SoccerMaster
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--siglip2_path", default="google/siglip2-large-patch16-512")
    parser.add_argument("--input_size", type=int, default=512)
    # Baselines
    parser.add_argument("--checkpoint", default=None, help="Override the pinned Hub checkpoint.")
    parser.add_argument("--revision", default=None)
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
        print(f"Wrote {os.path.join(args.out_dir, 'summary_all.md')}")
        return

    label = args.label or args.model
    settings = {"groups": args.groups, "ranking": args.ranking, "fill": args.fill, "map_key": args.map_key,
                "random_seeds": args.random_seeds}

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

    from dataset.video_utils_siglip import read_frames_decord
    from grounding_metrics import load_saliency_dir  # needs cv2; only the model run does

    test_config = load_config(args.config_path)["dataset"]["test"]
    keywords = test_config["keywords"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = {clip_id: (maps, meta) for clip_id, maps, meta in load_saliency_dir(args.saliency_dir, args.map_key)}
    print(f"[{label}] {len(saved)} saved maps in {args.saliency_dir} (key {args.map_key!r})")

    predictor = build_predictor(args, keywords, device, load_config(args.config_path))
    logit_scale = args.logit_scale if args.logit_scale is not None else predictor.logit_scale
    settings.update({"logit_scale": logit_scale, "grid": predictor.grid, "model": args.model})
    print(f"[{label}] token grid {predictor.grid}x{predictor.grid}, softmax temperature {logit_scale:.3f}, "
          f"{forwards_per_clip(args.groups, args.random_seeds)} forwards per clip")

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
        maps, meta = saved[clip_id]
        if tuple(maps.shape[1:]) != (predictor.grid, predictor.grid):
            raise ValueError(f"{clip_id}: map grid {maps.shape[1:]} != model token grid {predictor.grid}")
        target = int(meta.get("target_index", keywords.index(meta.get("ground_truth_text", caption))))

        frames, frame_indices, _ = read_frames_decord(video_path, args.num_frames, test_config.get("sample", "middle"))
        if frames.shape[0] != maps.shape[0]:
            raise ValueError(f"{clip_id}: {frames.shape[0]} frames but map has {maps.shape[0]}")
        x = predictor.preprocess(frames)
        if x.shape[0] != maps.shape[0] or x.shape[-1] % predictor.grid:
            raise ValueError(f"{clip_id}: preprocessed input {tuple(x.shape)} does not match grid {predictor.grid}")

        seeds = [1000 * index + s for s in range(args.random_seeds)]
        result = score_clip(predictor, x, maps.astype(np.float64), target, args.groups, args.ranking, args.fill,
                            seeds, logit_scale)
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
        with open(result_path, "w") as f:
            json.dump(result, f)
        n_done += 1
        s = result["saliency"]
        print(f"  [{index + 1}] {clip_id}: target={keywords[target]!r} pred={pred_text!r}"
              f"{' (differs from saved run!)' if mismatch else ''} p={result['p_clean_target']:.3f} "
              f"SaCo={s['saco']:+.3f} (rand {np.mean([r['saco'] for r in result['random']]):+.3f}) "
              f"AOPC={s['morf']['aopc']:.3f} first-flip={s['morf']['first_flip_fraction']:.1f} "
              f"({time.time() - started:.0f}s)")
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
