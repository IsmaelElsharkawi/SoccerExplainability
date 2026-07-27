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
