# Evidence Tables at ee51a38 — companion notes

Notes for `evidence_tables_ee51a38.html`, the companion to `evidence_tables.html`
(see `EVIDENCE_TABLES_README.md`) for NeurIPS 2026 submission #1152. Where the
original page's "Open items" flagged that tables under the current (post-redefinition)
metric definitions hadn't been built yet, this page is that rebuild.

---

## 1. What is on the page

| Table | Rows | Column groups | Answers |
|---|---|---|---|
| 1 — Headline | metric (all 8) | model × cue tier | How does every current metric move across tiers, per model? |
| 2 — T-IoU sensitivity | threshold 0.50–0.95 | model | Is the fixed cutoff cherry-picked, temporal side? |
| 2b — S-IoU sensitivity | threshold 0.50–0.95 | model | Same question, spatial side — recomputed from raw saliency maps (see §3). |
| 3 — Per-class | event class | metric × model | Which classes are hard, and for whom? |
| 4 — Bootstrap CI | metric (all 8) | model × sample size | How much of the spread is small-benchmark noise? |
| 5 — Sample size | metric (all 8) | model × n | Is 200 clips enough? |

The difference from the original page: there S-AUC/T-AUC/T-AP/T-IoU (sweep) were
marked superseded and held out of Tables 1–3; here every metric is current
throughout, because all ten source runs postdate the redefinition entirely.

---

## 2. Provenance

Source: `SoccerExplainability-output-20260727T084020Z-1-001.zip`, pulled from the
Drive output root (`SoccerExplainability-output/<experiment>/run_XXXX/`). All 10
runs report `git_commit: ee51a38` in `run_info.txt`:

| Model | Experiment dir | Runs used |
|---|---|---|
| SigLIP | `chefer_siglip` | `run_0029`, `run_8206`, `run_9683` |
| MatchVision | `chefer_matchvision_temporal` | `run_1581`, `run_2466`, `run_4885` |
| SoccerMaster | `chefer_soccermaster_temporal` | `run_0516`, `run_5874`, `run_6218`, `run_6780` |

Before computing anything:

- **Repeat-run checksums.** Each run's per-clip `(clip_id, prediction_text, summary)`
  content was hashed; runs within a model are byte-identical, confirming the
  deterministic-pipeline finding from the original page still holds (now for all
  three models, not just two — MatchVision has three completed runs here, not one).
  One representative run per model (`run_0029` / `run_2466` / `run_6218`) was used
  for every table; there is nothing to average.
- **Headline reproduction.** The P+S+C values reproduce the previously-verified
  numbers exactly — SigLIP 22.71 / 27.56 / 2.98 / 28.67, MatchVision 39.11 / 44.20 /
  7.47 / 14.99, SoccerMaster 37.05 / 50.47 / 3.62 / 14.90 (Energy / Pointing / S-IoU
  / T-IoU) — confirming this zip is the same lineage as the original page's numbers,
  not a divergent or broken-weights run.
- **Prediction-collapse check.** Per the original README's open item ("checking the
  prediction distribution is how a broken-weights run was caught previously"):
  SigLIP's `prediction_text` collapses to `foul lead to penalty` on 168/200 clips
  (5 distinct predictions, 7.5% accuracy); MatchVision and SoccerMaster show healthy
  spread (16 distinct predictions each, 62.5% / 66.0% accuracy). This is not a new
  problem introduced by this zip — the checkpoint reproduces the already-cited
  numbers exactly — but it means SigLIP's Energy/Pointing/IoU scores describe
  grounding for a mostly-wrong predicted label, not the true event class.

### Regenerating

Tables 1, 3, 4 and 5 come straight from `convergence_analysis.py`, run once per
model at the P+S+C tier (Table 1 additionally needs the P and P+S tiers):

```bash
python inference/convergence_analysis.py \
    --saliency_dir <run>/saliency \
    --selected_videos_json train_data/json/selected_videos_for_annotations.json \
    --tier small_large_visual_cues
```

Table 2 (T-IoU sensitivity) is a direct per-clip average of the `tIoU_sweep.per_threshold`
field already present in each sidecar's `summary.temporal_localization.tiers.small_large_visual_cues`
— no separate script; see the original README for why this grid moved from
0.1–0.9 to 0.50–0.95 at `ee51a38`.

Table 2b (S-IoU sensitivity) has no sidecar equivalent — S-AUC is the integral of
this exact curve (`coco_attribution_eval._spatial_auc`), but only the integrated
scalar is persisted, not the curve behind it. It is recomputed from the raw
per-frame 14×14 saliency maps saved in each run's `.npz` sidecars:

```bash
python inference/spatial_sensitivity_sweep.py \
    --saliency_dir <run>/saliency \
    --annotations_coco_json annotations-coco.json
```

Integrating this curve trapezoidally lands within 1–4% relative of Table 4's
S-AUC, not exact: S-AUC excludes individual frames whose rasterized 14×14 ROI
mask is degenerate (all-foreground or all-background) before integrating
per-frame, while this script averages IoU across frames first and integrates
that average — a real methodological difference in frame inclusion, not a bug.

---

## 3. Which metrics are current

All eight — Energy, Pointing, S-IoU, S-AUC, T-IoU, T-IoU (sweep), T-AUC, T-AP —
are current throughout this page. There is no superseded/current split to track
here, unlike the original page: every run behind these numbers was generated at
`ee51a38`, after the redefinition, so there are no legacy values to strike through.

---

## 4. Open items

- **Per-class counts shift slightly from the original page** (e.g. goal 24→21,
  corner 23→24, penalty 12→14) despite both totalling 200. 7/200 clips carry an
  overlapping-timestamp alternate label (e.g. a clip at `2_02_39.mp4` and a second
  at `2_02_39#corner.mp4` a few seconds apart); the `video → caption` lookup used
  by `convergence_analysis.py` doesn't have entries for the alternate-label variant,
  so those 7 clips fall back to the sidecar's own `ground_truth_text`. This zip drew
  a different overlap resolution than the run behind the original page, not a
  different benchmark.
- **Chefer-Spatial vs Chefer-Temporal** is still missing, same as the original page.
- **Run-to-run error bars are still not available** — the pipeline remains
  deterministic (see §2), so the measured run-to-run spread is exactly zero by
  construction, same as before. The bootstrap intervals in Table 4/5 remain the
  right instrument for a defensible uncertainty figure.
- **S-IoU sensitivity (Table 2b) doesn't reconcile exactly with S-AUC** — see the
  frame-inclusion note in §2. Close enough (1–4% relative) to trust the shape of
  the curve; not close enough to treat the trapezoidal integral of Table 2b as a
  drop-in replacement for Table 4's S-AUC column.

---

## 5. Editing the page

Same conventions as `evidence_tables.html` (see `EVIDENCE_TABLES_README.md` §5):
self-contained HTML, no external fonts/scripts, colours as CSS custom properties
redefined for dark mode under both `prefers-color-scheme` and
`:root[data-theme="dark"]`. This page reuses that exact style block rather than
inventing a new one, since the two are meant to be read as companions.

Table markup is generated rather than hand-written — the same rule applies here:
to change numbers, rerun the generation steps above against fresh sidecars rather
than editing cells by hand.
