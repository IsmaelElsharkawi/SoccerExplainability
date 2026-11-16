# Joint-ST

Joint-ST propagates the target-logit gradient through captured attention as
`C = mean_heads(max(gradient * attention, 0))`, retaining each self-attention
identity residual. It returns one raw, nonnegative `[T,H,W]` relevance volume.
It does not normalize individual frames or apply a normalized temporal scalar
after propagation. The factorized computation avoids constructing a dense
`(T*N) x (T*N)` graph.

The integrations follow each model's attention order:

| Model | Relevance path, from output to input |
| --- | --- |
| MatchVision, SoccerMaster | Classifier temporal readout, spatial pooling query, then spatial and temporal transitions in reverse backbone order |
| PE-AV | Temporal CLS readout, per-frame pooling query, spatial backbone |
| VideoPrism | Global pooling query, joint auxiliary layers, per-patch temporal layers, spatial layers |
| CLIP | Mean-pooled frame CLS seeds, independent spatial layers |
| SigLIP | Mean-pooled frame pooling queries, independent spatial layers |

CLIP and SigLIP have no cross-frame attention. Their Joint-ST limits seed each
frame by `1/T`; raw maps therefore differ from unscaled spatial Chefer by
that factor, up to floating-point rounding. No temporal edges are invented.

## Run

From the repository root, run one model with the existing test configuration:

```bash
python inference/run_joint_st.py \
  --model matchvision \
  --config_path config/pretrain_classification_cluster.py \
  --checkpoint /path/to/pretrained_classification.pth \
  --output_dir output_joint_st/matchvision
```

`--model` accepts `matchvision`, `soccermaster`, `clip`, `siglip`, `peav`, and
`videoprism`. SoccerMaster requires `--checkpoint` pointing to its directory
containing `backbone.pt` and `CaptionClassification.pt`. The four zero-shot
models load their released Hugging Face checkpoints; no `--checkpoint` is
needed. `--model_name` can override their model ID/path. PE-AV and VideoPrism
also accept `--revision`.

The configuration supplies the test manifest, video roots, class keywords,
and sampling mode. To use a different manifest, pass both `--test_json` and
`--video_base`. The manifest is the existing list format: `video`, `caption`,
and optional `variant`. Each caption must match a configured class keyword.
The runner always explains that ground-truth class. Decode failures stop the
run rather than substituting another example.

The default is 30 frames on `cuda:0`. Use `--max_clips 1` for a smoke run.
CLIP uses its released resize/center-crop processor; MatchVision, SoccerMaster
and SigLIP retain the existing SigLIP preprocessing. SoccerMaster additionally
resizes to 512. PE-AV and VideoPrism use their adapters' full-frame preprocessing.
VideoPrism's released checkpoint has 16 temporal positions; running 30 frames
uses its temporal-position interpolation.

Use the repository environment for MatchVision, SoccerMaster, CLIP, and
SigLIP. The PE-AV/VideoPrism APIs require Transformers revision
`b6d5084fb4a5dd11e44005a5fa009e7943271090` and `timm==1.0.28`, alongside the
repository's video-decoding dependencies. Their adapters pin checkpoint
revisions in their model modules. Their default runtime dtype is `bfloat16`;
`--dtype float32` overrides it.

Each example produces an NPZ with raw `saliency_maps`, `frame_scores` (spatial
means), and decoded `frame_indices`, plus JSON identifying the example and
target. Maps remain on each model's input token grid. The runner does not
compute grounding or perturbation metrics.

The existing MatchVision, SoccerMaster, and SigLIP attribution scripts also
accept `--propagation_mode joint_spatiotemporal`; their existing defaults are
preserved.

## Tests

```bash
# Repository environment (MatchVision/SoccerMaster/CLIP/SigLIP):
python -m pytest tests/test_joint_spatiotemporal_chefer.py \
  tests/test_joint_st_frame_encoders.py -q

# PE-AV/VideoPrism environment:
python -m pytest tests/test_joint_st_video_foundation.py -q
```

Tests compare factorized propagation with a dense graph, verify retained
identity and raw scaling, exercise real tiny CLIP/SigLIP transformers, and
check the PE-AV and VideoPrism relevance paths and the heatmap-saving runner.
They require no pretrained downloads. The repository's CLIP/SigLIP classes
use the Transformers 4 API; run their tests in the repository environment,
separately from the newer PE-AV/VideoPrism environment.
