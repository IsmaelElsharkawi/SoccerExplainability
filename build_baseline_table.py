"""Build the baseline-results LaTeX table from per-experiment eval JSONs.

Reads chefer_*_eval_results.json / gradcam_*_eval_results.json under
output_*/ and prints a LaTeX table mirroring the three-tier label hierarchy
of the original baseline (Table 1), with an added Chefer-Temporal column.

Cells whose JSON cannot be located are rendered as "--". Edit CANDIDATES
below if your output directories are named differently.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = ["SigLIP", "MatchVision", "SoccerMaster"]
METHODS = ["Grad-CAM", "Chefer-Spatial", "Chefer-Temporal"]

TIERS = [
    ("P",     "small_only"),
    ("P+S",   "small_large"),
    ("P+S+C", "small_large_visual_cues"),
]

METRICS = [
    ("Energy",   "spatial",  "mean_energy_inside_bbox"),
    ("Pointing", "spatial",  "mean_pointing_accuracy"),
    ("S-IoU",    "spatial",  "mean_iou"),
    ("S-AUC",    "spatial",  "mean_s_auc"),
    ("T-IoU",    "temporal", "mean_tIoU"),
    ("T-IoU (sweep)", "temporal", "mean_tIoU_sweep"),
    ("T-AUC",    "temporal", "mean_tAUC"),
    ("T-AP",     "temporal", "mean_tAP"),
]

# Candidate (directory, json_filename) pairs to try for each (model, method).
# The first existing file wins. Add more candidates if your dirs are named
# differently. None means the cell is N/A by design (e.g. SigLIP has no
# temporal modules to wrap).
CANDIDATES = {
    ("SigLIP", "Grad-CAM"): [
        ("output_gradcam_siglip", "gradcam_siglip_eval_results.json"),
        ("output_gradcam_siglip", "eval_results.json"),
    ],
    ("SigLIP", "Chefer-Spatial"): [
        ("output_chefer_siglip", "chefer_siglip_eval_results.json"),
        ("output_chefer_siglip", "eval_results.json"),
    ],
    ("SigLIP", "Chefer-Temporal"): None,

    ("MatchVision", "Grad-CAM"): [
        ("output_gradcam_matchvision", "gradcam_matchvision_eval_results.json"),
        ("output_gradcam_matchvision", "eval_results.json"),
    ],
    ("MatchVision", "Chefer-Spatial"): [
        ("output_chefer_matchvision_spatial",      "chefer_matchvision_eval_results.json"),
        ("output_chefer_matchvision_spatial_only", "chefer_matchvision_eval_results.json"),
        ("output_chefer_matchvision",              "chefer_matchvision_eval_results.json"),
    ],
    ("MatchVision", "Chefer-Temporal"): [
        ("output_chefer_matchvision_temporal", "chefer_matchvision_temporal_eval_results.json"),
    ],

    ("SoccerMaster", "Grad-CAM"): [
        ("output_gradcam_soccermaster", "gradcam_soccermaster_eval_results.json"),
        ("output_gradcam_soccermaster", "eval_results.json"),
    ],
    ("SoccerMaster", "Chefer-Spatial"): [
        ("output_chefer_soccermaster_spatial",      "chefer_soccermaster_eval_results.json"),
        ("output_chefer_soccermaster_spatial_only", "chefer_soccermaster_eval_results.json"),
        ("output_chefer_soccermaster",              "chefer_soccermaster_eval_results.json"),
    ],
    ("SoccerMaster", "Chefer-Temporal"): [
        ("output_chefer_soccermaster_temporal", "chefer_soccermaster_temporal_eval_results.json"),
    ],
}


def load_eval(model, method):
    candidates = CANDIDATES.get((model, method))
    if candidates is None:
        return None
    for d, fname in candidates:
        path = os.path.join(REPO_ROOT, d, fname)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return None


def lookup_metric(data, metric_kind, metric_key, tier_key):
    if data is None:
        return None
    try:
        if metric_kind == "spatial":
            return data["global_label_group_summary"][tier_key][metric_key]
        elif metric_kind == "temporal":
            return data["global_temporal_localization_summary"]["tiers"][tier_key][metric_key]
    except KeyError:
        return None
    return None


def fmt(value):
    if value is None:
        return "--"
    return f"{100 * value:.3f}"


def build_table():
    cache = {(m, mt): load_eval(m, mt) for m in MODELS for mt in METHODS}

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Baseline results for the three-tier label hierarchy across three explainability methods. Numbers are percentages. Tiers are abbreviated \textbf{P} (Primary Cues, \textit{small\_only}), \textbf{P+S} (Primary $+$ Secondary, \textit{small\_large}), and \textbf{P+S+C} (Primary $+$ Secondary $+$ Common, \textit{small\_large\_visual\_cues}). Cells marked ``--'' are unavailable: SigLIP has no temporal attention modules to wrap, hence no Chefer-Temporal variant.}")
    lines.append(r"    \label{tab:baseline_results_three_methods}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{ll ccc ccc ccc}")
    lines.append(r"        \toprule")
    lines.append(r"        & & \multicolumn{3}{c}{Grad-CAM} & \multicolumn{3}{c}{Chefer-Spatial} & \multicolumn{3}{c}{Chefer-Temporal} \\")
    lines.append(r"        \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}")
    lines.append(r"        \textbf{Model} & \textbf{Metric (\%)} & P & P+S & P+S+C & P & P+S & P+S+C & P & P+S & P+S+C \\")
    lines.append(r"        \midrule")

    for mi, model in enumerate(MODELS):
        for ri, (metric_label, metric_kind, metric_key) in enumerate(METRICS):
            if ri == 0:
                row_head = (
                    r"        \multirow{8}{*}{\textbf{" + model + r"}}"
                    r" & " + metric_label
                )
            else:
                row_head = r"            & " + metric_label
            cells = []
            for method in METHODS:
                data = cache[(model, method)]
                for tier_label, tier_key in TIERS:
                    cells.append(fmt(lookup_metric(data, metric_kind, metric_key, tier_key)))
            terminator = r" \\"
            lines.append(row_head + " & " + " & ".join(cells) + terminator)
        if mi < len(MODELS) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def report_coverage():
    print("# Coverage report (which cells have data)")
    for model in MODELS:
        for method in METHODS:
            data = load_eval(model, method)
            mark = "OK" if data is not None else (
                "N/A" if CANDIDATES.get((model, method)) is None else "MISSING"
            )
            print(f"  {model:13s}  {method:18s}  {mark}")
    print()


if __name__ == "__main__":
    report_coverage()
    print(build_table())
