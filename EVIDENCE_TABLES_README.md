# Consolidated Evidence Tables — companion notes

Notes for `evidence_tables.html`, the consolidated results page for NeurIPS 2026
submission #1152 (Evaluations & Datasets track).

The organising rule: **one table per metric or setting, all three model/method pairs
inside it.** The earlier page split every table three ways, one per model, which made
cross-model comparison a scrolling exercise. Here a reader answers "which model wins on
this metric" without leaving the table.

---

## 1. What is on the page

| Table | Rows | Column groups | Answers |
|---|---|---|---|
| 1 — Headline | metric | model × cue tier | How does each metric move across tiers, per model? |
| 2 — Sensitivity | threshold 0.1–0.9 | model | Is the fixed 0.5 cutoff cherry-picked? |
| 3 — Per-class | event class | metric × model | Which classes are hard, and for whom? |
| 4 — Bootstrap CI | metric | model × sample size | How much of the spread is small-benchmark noise? |
| Superseded | metric | model | Audit trail for the four redefined metrics |

Table 2 shades cells on a blue→ember ramp normalised across the whole table, so
magnitude reads at a glance and the three models stay comparable by eye. The ramp is a
nod to how attribution heatmaps themselves are rendered.

---

## 2. Provenance

Every number is **generated from the per-clip artifacts, not transcribed.** Source is the
`saliency/*.json` sidecars of the after-fix run (`SoccerExplainability-output-2026-07-25-…-after-fix.zip`),
200/200 clips for each of SigLIP, MatchVision and SoccerMaster.

Verified on generation: the tier P+S+C headline reproduces the previously reported values
exactly — SigLIP 22.71 / 27.56 / 2.98 / 28.67, MatchVision 39.11 / 44.20 / 7.47 / 14.99,
SoccerMaster 37.05 / 50.47 / 3.62 / 14.90 (Energy / Pointing / S-IoU / T-IoU).

### Regenerating

```bash
python inference/convergence_analysis.py \
    --saliency_dir <run>/saliency \
    --selected_videos_json train_data/json/selected_videos_for_annotations.json
```

That emits `convergence_summary.csv`, `convergence_curves.csv` and
`per_class_metrics.csv` under `<run>/saliency/analysis/`, which are the inputs behind
Tables 3 and 4.

---

## 3. Which metrics are current

**Current — safe to cite.** Energy, Pointing, S-IoU, T-IoU. These are the single-cutoff
metrics and were untouched by the metric redefinition.

**Superseded — do not cite.** S-AUC, T-AUC, T-AP, T-IoU (sweep). Commit `ee51a38`
redefined all four: they are now integrals of the IoU-vs-threshold curve rather than
rank-based ROC/PR scores. Two consequences that matter more than the numbers moving:

- **The scale changed.** These are IoU integrals now, not ranking probabilities. Old
  S-AUC ran 62–80 and old T-AUC 41–49; the new ones land where S-IoU and T-IoU live.
  **0.5 is no longer chance**, so every "well above chance" phrasing is void.
- **No old-vs-new comparison is meaningful**, even directionally.

The page shows the old values struck through, purely so the change is auditable.

`ee51a38` also moved the shared threshold sweep from 0.1–0.9 to
0.50, 0.55, … 0.95, so Table 2's grid is the pre-change one.

---

## 4. Open items

- **Tables under the new metric definitions are not built yet.** Runs at `ee51a38` exist
  in Drive under `SoccerExplainability-output/<experiment>/run_XXXX/`, but the eval JSONs
  are ~5 MB each and were not pulled for this page. Rebuilding Tables 1–4 against them is
  the next step.
- **Chefer-Spatial vs Chefer-Temporal** is still missing. Only temporal runs exist for
  MatchVision and SoccerMaster; `ismael/run_chefer_matchvision.sh` and
  `ismael/run_chefer_soccermaster.sh` would produce the counterpart.
- **Run-to-run error bars are still not available.** Repeat runs so far have produced
  byte-identical eval JSONs, so the measured variance is zero by construction. Genuine
  replicates need a varying seed or non-deterministic kernels left enabled.
- **Accuracy is not a pipeline field.** It has to be derived by comparing
  `prediction_text` against `ground_truth_text` in the sidecars. Checking the *prediction
  distribution* at the same time is worthwhile — a collapsed model shows one or two
  distinct predictions across all 200 clips, which is how a broken-weights run was caught
  previously.

---

## 5. Editing the page

`evidence_tables.html` is self-contained: no external fonts, scripts, styles or images,
which the artifact CSP would block. Colours are CSS custom properties defined once on
`:root` and redefined for dark mode under both `prefers-color-scheme` and
`:root[data-theme="dark"]`, so the viewer's theme toggle wins in both directions.

Table markup is generated rather than hand-written. To change numbers, rerun the
generation step against fresh sidecars rather than editing cells by hand — hand-editing is
how transcription errors enter, and the whole point of the provenance chain above is that
no figure on the page was typed by a human.


| Metric | SigLIP 50 | SigLIP 100 | SigLIP 150 | SigLIP 200 | MatchVision 50 | MatchVision 100 | MatchVision 150 | MatchVision 200 | SoccerMaster 50 | SoccerMaster 100 | SoccerMaster 150 | SoccerMaster 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Energy | 4.27 | 3.06 | 2.44 | 2.15 | 5.37 | 3.85 | 3.10 | 2.76 | 5.04 | 3.79 | 3.05 | 2.76 |
| Pointing | 7.63 | 5.55 | 4.30 | 3.96 | 6.62 | 4.33 | 3.94 | 3.13 | 7.42 | 5.09 | 4.13 | 3.43 |
| S-IoU | 0.92 | 0.64 | 0.53 | 0.47 | 1.80 | 1.32 | 1.02 | 0.91 | 0.85 | 0.59 | 0.49 | 0.40 |
| S-AUC | 0.79 | 0.56 | 0.46 | 0.41 | 1.86 | 1.22 | 1.07 | 0.92 | 0.63 | 0.42 | 0.37 | 0.33 |
| T-IoU | 4.28 | 3.20 | 2.52 | 2.17 | 3.33 | 2.38 | 2.02 | 1.71 | 4.45 | 3.14 | 2.64 | 2.20 |
| T-IoU (sweep) | 3.07 | 2.21 | 1.74 | 1.49 | 2.26 | 1.57 | 1.27 | 1.16 | 2.63 | 1.88 | 1.54 | 1.34 |
| T-AUC | 3.17 | 2.21 | 1.77 | 1.47 | 2.32 | 1.69 | 1.30 | 1.10 | 2.79 | 1.86 | 1.67 | 1.41 |
| T-AP | 3.21 | 2.31 | 1.77 | 1.57 | 2.57 | 1.81 | 1.49 | 1.24 | 3.17 | 2.33 | 1.85 | 1.65 |