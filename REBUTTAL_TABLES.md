# SoccerLens — Rebuttal Evidence Tables

NeurIPS 2026 Evaluations and Datasets Track, Submission #1152. Compiled from the
`run_chefer_siglip.sh` / `run_chefer_matchvision_temporal.sh` /
`run_chefer_soccermaster_temporal.sh` runs executed 2026-07-25 (`SoccerLens.ipynb`,
branch `Rebuttal`). All three runs cover the full 200-clip benchmark; no clips were
dropped in any table below. Full derivation notes are in the "Provenance & caveats"
section at the end — read that before citing a number in the rebuttal response.

> ⚠️ **STALE — S-AUC, T-AUC and T-AP columns must be regenerated; Table 6 is retired
> (2026-07-26).** All three metrics were redefined in `coco_attribution_eval.py` after this
> document was compiled: they are now integrals of the **IoU-vs-threshold curve** over the
> sweep **0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95**, not rank-based ROC/PR
> scores. They are now on the same 0–1 scale as IoU itself, so **0.5 is no longer chance**
> and the old values are not comparable to the new ones even qualitatively. The T-IoU sweep
> moved to that same range, so **Table 5 is stale too**. The separate S-IoU sweep has been
> removed from the pipeline, retiring Table 6. Energy, Pointing, S-IoU and T-IoU (single
> fixed 0.5 cutoff) are unaffected. See §1 for the new definitions and §9 for the fallout.

> **SoccerMaster figures updated 2026-07-26.** All SoccerMaster numbers in every table
> below come from the re-run in `SoccerExplainability-output-2026-07-25-...-after-fix.zip`,
> which corrected a mis-configured model directory in the original run. The earlier
> SoccerMaster figures were produced by a model that had not loaded its weights correctly
> and should not be cited (they understated Pointing by ~40pp and S-AUC by ~15pp). SigLIP
> and MatchVision are unchanged from the original bundle.

Three model/method combinations are covered:

| Model | Method | Role |
|---|---|---|
| SigLIP2-large | Chefer-Spatial | Generic (non-soccer-specialized) baseline; no temporal attention to wrap |
| MatchVision | Chefer-Temporal | Soccer-specialized, architecture family 1 |
| SoccerMaster | Chefer-Temporal | Soccer-specialized, architecture family 2 |

---

## 1. Metric glossary

All metrics are computed per annotated frame, then averaged first within a clip and
then across clips (macro-average) — never pooled across all frames directly. This
matches how the pipeline (`inference/coco_attribution_eval.py`) aggregates internally.

### Cue tiers

The benchmark's three-level annotation scheme. Tiers are cumulative — each one adds a
category of ground-truth box on top of the last:

| Tier | COCO categories included | Meaning |
|---|---|---|
| **P** (`small_only`) | small label | Primary cues only — the most direct, event-defining visual evidence |
| **P+S** (`small_large`) | small label + large label | Primary + Secondary — adds larger supporting regions |
| **P+S+C** (`small_large_visual_cues`) | small label + large label + visual cue | Primary + Secondary + Common — broadest tier; **this is the tier used for the paper's headline numbers** and for Tables 3–4 below |

### Spatial metrics (per-frame heatmap vs. cue box)

| Metric | Threshold? | Definition |
|---|---|---|
| **Energy** | No | Fraction of the heatmap's total mass that falls inside the cue box(es): `sum(map · mask) / sum(map)`. |
| **Pointing** | No (peak-only) | 1 if the single highest-attribution cell falls inside a cue box, else 0. |
| **S-IoU** | Yes (`cam_threshold`, default 0.5) | Binarize the heatmap at `threshold × max`, then IoU against the cue-box mask. Depends on one arbitrary cutoff — this is exactly what reviewers fUPK and PBFf flagged. |
| **S-AUC** | Yes — the shared sweep | S-IoU recomputed at every ratio in `THRESHOLD_SWEEP`, then the area under that IoU-vs-threshold curve, normalized by the threshold range. Answers fUPK's concern by scoring across a range of cutoffs instead of one. **On the IoU scale, not a probability** — it replaces the separate S-IoU sweep rather than complementing it. |

### Temporal metrics (per-frame salience-over-time vs. annotated frames)

Per-frame heatmap intensity is averaged spatially to get one salience score per frame
(30 scores per clip), which is then compared against the set of frames that carry a
cue-box annotation.

| Metric | Threshold? | Definition |
|---|---|---|
| **T-IoU** | Yes (same ratio as S-IoU, default 0.5) | Binarize the per-frame salience signal at `ratio × max` to get a predicted "salient frames" set; IoU against the ground-truth annotated-frames set. |
| **T-IoU (sweep)** | Yes — the shared sweep | Mean T-IoU over the sweep ratios. |
| **T-AUC** | Yes — the shared sweep | Area under the T-IoU-vs-threshold curve, normalized by the threshold range. The temporal counterpart of S-AUC. |
| **T-AP** | Yes — the shared sweep | The same per-threshold T-IoU values integrated average-precision style: each value weighted by the recall increment its threshold step contributes, divided by the recall range covered. Emphasises the thresholds that actually recover annotated frames. |

**The shared threshold sweep.** Every threshold-dependent metric now uses one range,
`THRESHOLD_SWEEP` in `inference/coco_attribution_eval.py`:

    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95

as a ratio of the per-frame max heatmap value (spatial) or per-clip max frame score
(temporal). It starts at 0.5 because a cutoff below half of max admits the method's own
noise floor, so operating points down there describe noise ranking rather than grounding.

**Scale change — read this before writing any prose around these numbers.** S-AUC, T-AUC
and T-AP are now IoU integrals, not ranking probabilities. A value of 0.5 is no longer
"chance"; these are absolute overlap scores and will land in the same low range as S-IoU
and T-IoU. Any sentence of the form "S-AUC is stable at 62–80%, well above chance" is void.

### Bootstrap confidence intervals (Table 3)

1000-resample percentile bootstrap over the 200 clips (`inference/convergence_analysis.py`).
"Converges at n=" is the smallest prefix of clips (in dataset order) at which the running
CI half-width first drops to ≤ 2.0 percentage points and stays there (minimum n=5 to avoid
degenerately tight small-sample CIs); "not within 200" means it never did.

---

## 2. Table 1 — Headline results (tier P+S+C)

| Model | Method | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|
| SigLIP | Chefer-Spatial | 22.71 | 27.56 | 2.98 | 62.03 | 28.67 | 25.71 | 49.25 | 44.13 |
| MatchVision | Chefer-Temporal | 39.11 | 44.20 | 7.47 | 78.77 | 14.99 | 17.11 | 48.11 | 43.37 |
| SoccerMaster | Chefer-Temporal | 37.05 | 50.47 | 3.62 | 73.37 | 14.90 | 15.92 | 41.58 | 40.34 |

The two soccer-specialised models lead spatial grounding — MatchVision at S-AUC 78.8 and
SoccerMaster at 73.4, against 62.0 for the generic SigLIP baseline. The ordering inverts on
the temporal side: SigLIP posts the highest T-IoU (28.67) and SoccerMaster the lowest
(14.90). Spatial and temporal grounding do not move together across models.

---

## 3. Table 2 — Full cue-tier breakdown

**SigLIP (Chefer-Spatial)**

| Tier | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|
| P | 16.50 | 24.59 | 3.47 | 63.22 | 19.27 | 17.75 | 47.56 | 32.27 |
| P+S | 22.04 | 30.14 | 2.89 | 61.56 | 19.34 | 17.85 | 47.62 | 32.44 |
| P+S+C | 22.71 | 27.56 | 2.98 | 62.03 | 28.67 | 25.71 | 49.25 | 44.13 |

**MatchVision (Chefer-Temporal)**

| Tier | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|
| P | 26.20 | 30.35 | 6.90 | 80.34 | 11.72 | 12.68 | 48.84 | 31.52 |
| P+S | 34.68 | 38.97 | 6.04 | 77.29 | 11.75 | 12.77 | 48.99 | 31.73 |
| P+S+C | 39.11 | 44.20 | 7.47 | 78.77 | 14.99 | 17.11 | 48.11 | 43.37 |

**SoccerMaster (Chefer-Temporal)**

| Tier | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|
| P | 24.33 | 35.90 | 4.25 | 75.34 | 8.07 | 9.38 | 36.13 | 26.66 |
| P+S | 34.24 | 47.88 | 3.74 | 75.59 | 8.12 | 9.45 | 36.08 | 26.83 |
| P+S+C | 37.05 | 50.47 | 3.62 | 73.37 | 14.90 | 15.92 | 41.58 | 40.34 |

---

## 4. Table 3 — Bootstrap 95% confidence intervals (tier P+S+C)

| Model | Metric | Mean (%) | 95% CI | Half-width | Converges at n= |
|---|---|---|---|---|---|
| SigLIP | Energy | 22.71 | [20.65, 24.95] | ±2.15 | not within 200 |
| SigLIP | Pointing | 27.56 | [24.02, 31.60] | ±3.79 | not within 200 |
| SigLIP | S-IoU | 2.98 | [2.52, 3.48] | ±0.48 | 5 |
| SigLIP | S-AUC | 62.03 | [60.83, 63.34] | ±1.25 | 5 |
| SigLIP | T-IoU | 28.67 | [26.62, 30.83] | ±2.10 | 194 |
| SigLIP | T-IoU (sweep) | 25.71 | [24.16, 27.35] | ±1.59 | 99 |
| SigLIP | T-AUC | 49.25 | [46.83, 51.73] | ±2.45 | not within 200 |
| SigLIP | T-AP | 44.13 | [41.42, 46.72] | ±2.65 | not within 200 |
| MatchVision | Energy | 39.11 | [36.44, 41.97] | ±2.76 | not within 200 |
| MatchVision | Pointing | 44.20 | [40.86, 47.54] | ±3.34 | not within 200 |
| MatchVision | S-IoU | 7.47 | [6.64, 8.40] | ±0.88 | 6 |
| MatchVision | S-AUC | 78.77 | [77.52, 79.97] | ±1.23 | 6 |
| MatchVision | T-IoU | 14.99 | [13.29, 16.64] | ±1.68 | 139 |
| MatchVision | T-IoU (sweep) | 17.11 | [15.90, 18.41] | ±1.25 | 54 |
| MatchVision | T-AUC | 48.11 | [45.77, 50.48] | ±2.35 | not within 200 |
| MatchVision | T-AP | 43.37 | [40.62, 46.11] | ±2.75 | not within 200 |
| SoccerMaster | Energy | 37.05 | [34.22, 39.75] | ±2.76 | not within 200 |
| SoccerMaster | Pointing | 50.47 | [46.72, 54.29] | ±3.78 | not within 200 |
| SoccerMaster | S-IoU | 3.62 | [3.25, 4.04] | ±0.40 | 5 |
| SoccerMaster | S-AUC | 73.37 | [71.33, 75.43] | ±2.05 | 172 |
| SoccerMaster | T-IoU | 14.90 | [12.61, 17.30] | ±2.35 | 194 |
| SoccerMaster | T-IoU (sweep) | 15.92 | [14.14, 17.64] | ±1.75 | 141 |
| SoccerMaster | T-AUC | 41.58 | [38.90, 44.37] | ±2.73 | not within 200 |
| SoccerMaster | T-AP | 40.34 | [37.22, 43.20] | ±2.99 | not within 200 |

Consistent pattern across all three models: S-IoU converges almost immediately (n=5–6
clips), and S-AUC does too for SigLIP and MatchVision (n=5–6), though SoccerMaster's S-AUC
needs n=172 — its per-clip spread is much wider. Energy, Pointing, T-AUC, and T-AP never
tighten below ±2pp even at the full n=200, for any model — worth stating plainly in the
rebuttal rather than glossing over.

---

## 5. Table 4 — Per-class metrics (tier P+S+C)

Sorted by n descending. ⚠ marks classes with n < 10, per PBFf's explicit caution against
strong per-class conclusions on those.

**SigLIP (Chefer-Spatial)**

| Event class | n | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|
| corner | 24 | 17.16 | 5.90 | 0.30 | 56.63 | 34.70 | 27.11 | 47.77 | 42.34 |
| foul with no card | 21 | 14.32 | 12.23 | 2.40 | 60.11 | 29.12 | 25.59 | 56.24 | 43.95 |
| goal | 21 | 39.50 | 56.49 | 3.28 | 57.36 | 29.62 | 28.71 | 42.35 | 48.32 |
| yellow card | 20 | 26.48 | 44.51 | 5.42 | 68.54 | 25.58 | 24.22 | 45.90 | 45.85 |
| lead to corner | 19 | 4.30 | 2.65 | 0.87 | 60.64 | 16.70 | 13.60 | 46.90 | 23.64 |
| free kick | 18 | 19.40 | 22.08 | 4.75 | 67.32 | 26.34 | 23.14 | 46.54 | 39.91 |
| substitution | 17 | 19.00 | 18.02 | 4.24 | 67.48 | 37.50 | 34.48 | 51.81 | 55.48 |
| injury | 16 | 32.47 | 30.47 | 2.88 | 63.27 | 32.05 | 29.57 | 61.19 | 54.74 |
| penalty | 14 | 45.27 | 68.94 | 3.92 | 58.56 | 29.15 | 27.51 | 44.71 | 45.91 |
| foul lead to penalty | 13 | 18.62 | 23.50 | 1.81 | 60.06 | 24.30 | 23.58 | 47.72 | 42.86 |
| second yellow card ⚠ | 7 | 19.69 | 36.94 | 5.35 | 67.71 | 33.34 | 28.70 | 50.01 | 49.98 |
| throw in ⚠ | 7 | 12.26 | 16.88 | 1.36 | 61.91 | 27.48 | 25.07 | 56.07 | 42.78 |
| red card ⚠ | 3 | 32.24 | 49.44 | 5.05 | 61.12 | 21.49 | 21.69 | 46.04 | 39.51 |

**MatchVision (Chefer-Temporal)**

| Event class | n | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|
| corner | 24 | 47.27 | 52.72 | 6.16 | 77.51 | 16.03 | 17.95 | 55.29 | 45.97 |
| foul with no card | 21 | 37.10 | 49.72 | 9.56 | 82.85 | 13.22 | 14.73 | 40.46 | 36.54 |
| goal | 21 | 54.71 | 56.44 | 4.91 | 76.64 | 15.90 | 20.55 | 53.06 | 56.38 |
| yellow card | 20 | 42.45 | 50.53 | 9.88 | 80.21 | 18.89 | 20.29 | 45.54 | 47.69 |
| lead to corner | 19 | 11.37 | 15.40 | 7.72 | 82.29 | 11.22 | 10.68 | 50.69 | 24.54 |
| free kick | 18 | 34.03 | 38.51 | 5.93 | 76.35 | 15.35 | 15.83 | 47.17 | 40.97 |
| substitution | 17 | 41.03 | 47.04 | 12.65 | 77.68 | 8.25 | 12.63 | 33.22 | 39.49 |
| injury | 16 | 46.40 | 42.51 | 6.43 | 76.74 | 18.50 | 20.66 | 50.88 | 51.57 |
| penalty | 14 | 55.75 | 57.62 | 4.50 | 78.24 | 15.54 | 18.95 | 48.78 | 47.33 |
| foul lead to penalty | 13 | 28.64 | 34.00 | 6.74 | 78.65 | 17.39 | 19.45 | 55.85 | 46.02 |
| second yellow card ⚠ | 7 | 37.48 | 52.91 | 10.81 | 82.05 | 11.87 | 16.64 | 45.77 | 45.01 |
| throw in ⚠ | 7 | 14.29 | 13.95 | 3.84 | 76.53 | 16.37 | 17.47 | 53.84 | 38.90 |
| red card ⚠ | 3 | 42.43 | 51.11 | 6.54 | 74.40 | 19.31 | 18.06 | 40.88 | 39.17 |

**SoccerMaster (Chefer-Temporal)**

| Event class | n | Energy | Pointing | S-IoU | S-AUC | T-IoU | T-IoU(sweep) | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|
| corner | 24 | 48.59 | 60.69 | 1.80 | 71.99 | 8.72 | 11.55 | 37.47 | 36.42 |
| foul with no card | 21 | 36.52 | 62.53 | 4.60 | 85.51 | 9.76 | 11.55 | 36.78 | 33.54 |
| goal | 21 | 47.79 | 55.44 | 2.83 | 64.10 | 25.80 | 25.76 | 55.61 | 60.00 |
| yellow card | 20 | 35.24 | 57.04 | 4.41 | 77.57 | 15.93 | 17.43 | 43.16 | 45.35 |
| lead to corner | 19 | 14.79 | 29.43 | 5.28 | 78.84 | 4.45 | 5.39 | 27.76 | 16.84 |
| free kick | 18 | 32.95 | 51.40 | 3.35 | 79.85 | 15.90 | 17.37 | 52.33 | 43.11 |
| substitution | 17 | 21.25 | 26.32 | 3.61 | 49.60 | 35.80 | 30.79 | 58.48 | 53.24 |
| injury | 16 | 53.06 | 63.38 | 3.88 | 77.64 | 16.50 | 17.93 | 45.03 | 45.91 |
| penalty | 14 | 44.64 | 56.82 | 2.92 | 61.96 | 15.05 | 16.12 | 28.33 | 38.83 |
| foul lead to penalty | 13 | 41.66 | 56.77 | 4.88 | 82.02 | 9.50 | 11.07 | 31.97 | 33.58 |
| second yellow card ⚠ | 7 | 35.67 | 40.15 | 4.86 | 80.62 | 4.43 | 9.83 | 28.61 | 38.48 |
| throw in ⚠ | 7 | 16.55 | 14.85 | 1.04 | 73.79 | 9.69 | 10.68 | 44.57 | 34.23 |
| red card ⚠ | 3 | 50.80 | 51.67 | 2.25 | 72.69 | 9.46 | 12.26 | 34.90 | 32.59 |

---

## 6. Table 5 — T-IoU threshold sensitivity

Directly from `global_temporal_localization_summary.tiers.*.mean_tIoU_per_threshold` —
computed natively by the pipeline (`TIOU_THRESHOLD_SWEEP` in `coco_attribution_eval.py`),
full n=200 for every model.

**SigLIP**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 24.65 | 24.34 | 23.53 | 21.92 | 19.27 | 17.24 | 13.06 | 9.51 | 6.21 | 47.56 | 32.27 |
| P+S | 24.87 | 24.54 | 23.73 | 21.99 | 19.34 | 17.30 | 13.10 | 9.60 | 6.21 | 47.62 | 32.44 |
| P+S+C | 36.80 | 36.34 | 35.12 | 32.79 | 28.67 | 24.29 | 17.93 | 12.06 | 7.42 | 49.25 | 44.13 |

**MatchVision**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 24.25 | 20.94 | 17.61 | 14.79 | 11.72 | 9.29 | 6.62 | 5.45 | 3.47 | 48.84 | 31.52 |
| P+S | 24.44 | 21.18 | 17.74 | 14.85 | 11.75 | 9.33 | 6.65 | 5.49 | 3.46 | 48.99 | 31.73 |
| P+S+C | 35.22 | 29.45 | 24.00 | 19.30 | 14.99 | 11.41 | 8.39 | 6.64 | 4.56 | 48.11 | 43.37 |

**SoccerMaster**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | T-AUC | T-AP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 20.01 | 16.60 | 13.19 | 10.15 | 8.07 | 5.80 | 4.76 | 3.15 | 2.70 | 36.13 | 26.66 |
| P+S | 20.15 | 16.65 | 13.25 | 10.24 | 8.12 | 5.88 | 4.81 | 3.20 | 2.76 | 36.08 | 26.83 |
| P+S+C | 31.26 | 26.79 | 22.63 | 18.62 | 14.90 | 11.34 | 8.25 | 5.63 | 3.88 | 41.58 | 40.34 |

T-IoU degrades smoothly with no cliff at 0.5, and the threshold-free T-AUC/T-AP tell the
same qualitative story — the fixed cutoff isn't cherry-picked to flatter or penalize any
model.

## 7. Table 6 — S-IoU threshold sensitivity

**This sweep is reconstructed, not pipeline-native for these runs.** At the time all three
runs were executed, `coco_attribution_eval.py` only binarized spatial heatmaps at the fixed
`cam_threshold=0.5` — there was no spatial equivalent of `TIOU_THRESHOLD_SWEEP`. (A
`_spatial_iou_sweep` helper has since been added to the file, so future runs will emit
`iou_sweep` natively; it post-dates this bundle.) These numbers were reconstructed by
re-running the exact same IoU logic (`cam_binary = attribution_map >= ratio · max`, same
intersection/union, masks built at the heatmap's own resolution) against the raw per-frame
`saliency_maps.npz` heatmaps and `annotations-coco.json` boxes, at ratios 0.1–0.9, on the
full n=200 clips for every model. Validated against production: the 0.5-ratio column
reproduces the official `mean_iou` exactly for all three models (SigLIP P+S+C: 2.9769%
reconstructed vs. 2.977% official; MatchVision P+S+C: 7.4683% vs. 7.468%; SoccerMaster
P+S+C: 3.6200% vs. 3.620%).

**SigLIP**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | S-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 7.98 | 5.91 | 4.84 | 3.95 | 3.47 | 2.92 | 2.42 | 2.10 | 1.79 | 63.22 |
| P+S | 9.39 | 6.07 | 4.56 | 3.43 | 2.89 | 2.33 | 1.97 | 1.67 | 1.41 | 61.56 |
| P+S+C | 9.63 | 6.17 | 4.61 | 3.53 | 2.98 | 2.48 | 2.07 | 1.75 | 1.50 | 62.03 |

**MatchVision**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | S-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 11.99 | 9.97 | 8.74 | 7.76 | 6.90 | 6.08 | 5.57 | 5.07 | 4.74 | 80.34 |
| P+S | 12.44 | 9.64 | 8.12 | 7.03 | 6.04 | 5.11 | 4.69 | 4.21 | 3.94 | 77.29 |
| P+S+C | 14.96 | 11.93 | 9.93 | 8.51 | 7.47 | 6.59 | 6.04 | 5.57 | 5.11 | 78.77 |

**SoccerMaster**

| Tier | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | S-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 12.70 | 9.69 | 7.43 | 5.55 | 4.25 | 3.28 | 2.44 | 2.00 | 1.67 | 75.34 |
| P+S | 14.14 | 9.87 | 7.12 | 5.08 | 3.74 | 2.82 | 2.11 | 1.67 | 1.38 | 75.59 |
| P+S+C | 15.47 | 10.37 | 7.20 | 5.02 | 3.62 | 2.71 | 2.09 | 1.67 | 1.41 | 73.37 |

**This remains the rebuttal's core threshold-sensitivity evidence, but the effect is
model-dependent** — worth stating precisely rather than over-claiming. Across the 0.1 → 0.9
sweep at tier P+S+C:

| Model | S-IoU fall | T-IoU fall | Steeper metric |
|---|---|---|---|
| SigLIP | 9.63 → 1.50 (6.4×) | 36.80 → 7.42 (5.0×) | S-IoU |
| MatchVision | 14.96 → 5.11 (2.9×) | 35.22 → 4.56 (7.7×) | T-IoU |
| SoccerMaster | 15.47 → 1.41 (11.0×) | 31.26 → 3.88 (8.1×) | S-IoU |

S-IoU is the more threshold-sensitive metric for two of the three models, and dramatically
so for SoccerMaster; for MatchVision the ordering reverses. What holds across *every*
model and tier is the positive claim: S-IoU is low in absolute terms and swings by
3–11× depending purely on where the cutoff is placed, whereas the threshold-free S-AUC is
stable at 62–80% throughout. That is why S-AUC is the more defensible headline spatial
metric, and it directly answers fUPK's concern that "the 50%-of-max threshold may
over-penalize."

---

## 8. Reviewer-concern coverage map

| Concern | Raised by | Status | Evidence |
|---|---|---|---|
| Confidence intervals | PBFf W7 | **Answered** | Table 3 |
| Per-class analysis / small-class caution | PBFf W1, Q1; fUPK W1; BYLd Q2 | **Answered** | Table 4 (n column exposes imbalance directly) |
| T-IoU threshold sensitivity | PBFf W4 | **Answered** | Table 5 |
| Threshold-free alternative to IoU (S-AUC) | fUPK W3, Q | **Answered** | Tables 1–4, 6 (S-AUC column throughout) |
| S-IoU threshold sensitivity | fUPK Q (implied by W3) | **Answered** | Table 6 |
| Grounding scores conditioned on correct predictions | PBFf W7 | **Now tractable** | SoccerMaster 66.0% and MatchVision 62.5% exact-match accuracy support the split; SigLIP's 7.5% still unexplained. See caveat below |
| Chefer-T benefit over Chefer-Spatial (per-event) | PBFf W3; BYLd O1 | **Open** | Spatial-only scripts exist (`run_chefer_matchvision.sh`, `run_chefer_soccermaster.sh`) but haven't been run |
| Validation via alternative/perturbation-based attribution (RISE etc.) | PBFf W8; fUPK W2, Q | **Open** | No work done |
| More generic SOTA VLMs (Qwen, InternVL, etc.) | PBFf W5; fUPK W4 | **Open** | No work done |
| Single broadcast source | PBFf W6 | **Open** | No work done |
| Annotator qualifications / IAA / quality control | PBFf W2; BYLd C1 | **Open** | Not an evaluation-data question — needs input from whoever ran annotation |
| Fast vs. slow events, cancelled-event analysis | BYLd Q1 | **Open** | No work done |

---

## 9. Provenance & caveats

- **Source of truth**: `chefer_siglip_eval_results.json`, `chefer_matchvision_temporal_eval_results.json`,
  `chefer_soccermaster_temporal_eval_results.json`, and each run's
  `saliency/analysis/{convergence_summary.csv,per_class_metrics.csv}`. SigLIP and
  MatchVision come from `SoccerExplainability-output-2026-07-25-001.zip`; **SoccerMaster
  comes from the later `...-after-fix.zip` bundle** (model-directory fix, re-run
  2026-07-25 13:51 vs. 06:23/06:50 for the other two). Both zips contain the complete
  200/200 per-clip artifacts — an earlier, separately-downloaded copy of the output folder
  was missing ~20 files per model due to an incomplete Drive→Downloads export, since
  corrected by re-extracting the zip. Clip count verified at 200 for the SoccerMaster
  re-run (`len(per_video) == 200`).
- **Cross-checked** against the full `SoccerLens.ipynb` run logs (same three cells,
  `!sh run_chefer_siglip.sh` / `run_chefer_matchvision_temporal.sh` /
  `run_chefer_soccermaster_temporal.sh`) — every number in Tables 1–5 matches the printed
  console summaries exactly. No errors, crashes, or NaN-propagation issues in any run;
  only benign pip/dependency and deprecation warnings during environment setup.
- **S-AUC / T-AUC / T-AP were redefined on 2026-07-26, after every run in this bundle**, and
  the T-IoU sweep range changed with them. Previously all three called
  `sklearn.roc_auc_score` / `average_precision_score`, which rank pixels (or frames) and
  sweep every score down to ~0. They are now integrals of the IoU-vs-threshold curve over
  the shared sweep (see §1). Consequences when the runs are redone:
  - **The scale changed.** These are IoU integrals now, not ranking probabilities. Old
    S-AUC ~62–80 and T-AUC ~41–49 will drop into the range S-IoU and T-IoU occupy. This is
    a change of units, not a regression — but no old-vs-new comparison is meaningful, and
    every "above chance" phrasing has to go.
  - **Tables 1–5 all need regenerating**, not just the AUC/AP columns: Table 5's T-IoU
    sweep moved from 0.1–0.9 to the new range, so its columns change too. Only Energy,
    Pointing, S-IoU and single-cutoff T-IoU carry over unchanged.
  - **Table 6 is retired.** The separate S-IoU sweep was removed from the pipeline at the
    same time — S-AUC now *is* the integrated S-IoU sweep, so reporting both is redundant.
    `mean_iou_sweep` / `mean_iou_per_threshold` no longer exist in the output JSON, and
    `convergence_analysis.py` no longer emits an "S-IoU (sweep)" column.
  - **This weakens the fUPK W3 answer and that needs a decision.** S-AUC was added
    specifically as a *threshold-free* alternative to S-IoU; it is now threshold-dependent
    and on the IoU scale, so the paper no longer has a threshold-free spatial metric.
    Energy and Pointing remain genuinely threshold-free and should carry that argument
    instead.
  - **T-AUC is now nearly identical to T-IoU (sweep)** — both integrate the same
    per-threshold T-IoU values, differing only in trapezoidal vs. arithmetic weighting. On
    random-map spot checks they agree to within 0.001–0.011. Reporting both as separate
    columns will invite a reviewer question; consider dropping one.
- **Chefer-Spatial vs. Chefer-Temporal comparison is still missing.** This bundle only
  has Chefer-Temporal runs for MatchVision/SoccerMaster; no spatial-only counterpart, so
  the per-model delta table that would directly answer PBFf W3 / BYLd O1 can't be built
  yet. The spatial-only scripts already exist in the repo
  (`run_chefer_matchvision.sh`, `run_chefer_soccermaster.sh`) — running those would
  produce the missing data.
- **"Grounding scores conditioned on correct predictions" (PBFf W7) is now tractable for
  SoccerMaster.** The previously reported 3.5% (7/200) exact-match accuracy was an artefact
  of the mis-configured model directory, not of the label taxonomy. On the after-fix run
  SoccerMaster scores **66.0% (132/200)**, in line with MatchVision's 62.5% (125/200), and
  the residual errors are genuine semantic confusions (free kick → foul with no card ×6,
  second yellow card → yellow card ×5, penalty → goal ×4) rather than vocabulary
  mismatches. SigLIP's 7.5% (15/200) is still anomalous and remains unexplained. No
  `accuracy` field is computed anywhere in the pipeline, so these figures come from
  comparing the per-clip `prediction_text`/`ground_truth_text` fields in the saliency
  sidecars. The conditioned-on-correct grounding table can now be built for SoccerMaster
  and MatchVision; SigLIP would need its accuracy question resolved first.
