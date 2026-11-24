# The SoccerLens dataset

**Everything that makes up the SoccerLens benchmark is in this folder.** Nothing else in
the repository is part of the dataset; the rest is code.

```
data/
├── annotations/                                   <- THE BENCHMARK ANNOTATIONS (COCO format)
│   ├── annotations-coco.json                      200 clips, human annotations   (gold)
│   ├── annotations-coco-extra800.json             800 clips, Gemini annotations  (silver)
│   ├── annotations-coco-merged.json               all 1,000 clips (the two files above, merged)
│   └── merge_coco_validation.txt                  integrity checks of the merge
└── clips/                                         <- WHICH CLIPS ARE IN THE BENCHMARK
    ├── selected_videos_for_annotations.json       the 200 human-annotated clips
    ├── selected_videos_for_annotations_800.json   the 800 Gemini-annotated clips
    ├── selected_videos_for_annotations_merged.json all 1,000 clips
    └── one_video.json                             13 clips (one per class), for quick tests
```

| Split | Clips | Annotated frames | Boxes | Annotation file | Clip list |
|---|---|---|---|---|---|
| Human (gold) | 200 | 2,711 | 4,687 | `annotations/annotations-coco.json` | `clips/selected_videos_for_annotations.json` |
| Gemini (silver) | 800 | 23,997 | 24,442 | `annotations/annotations-coco-extra800.json` | `clips/selected_videos_for_annotations_800.json` |
| **All (1,000 clips)** | **1,000** | **26,708** | **29,129** | `annotations/annotations-coco-merged.json` | `clips/selected_videos_for_annotations_merged.json` |

The 1,000-clip results of the paper use the merged file. The 200-clip human file is the
default of every script.

## The video clips are not included

The annotations refer to 30-second broadcast clips from the SoccerNet videos (the
MatchTime test split), which we cannot redistribute. To obtain them:

1. Download the SoccerNet 720p match videos from <https://www.soccer-net.org/> (this
   requires signing the SoccerNet NDA).
2. Cut each clip from **15 s before to 15 s after** its timestamp. The clip name encodes
   the source: `<league>_<season>/<match>/<half>_<mm>_<ss>.mp4` is half `<half>` of
   `<league>/<season>/<match>/`, centred on minute `<mm>`, second `<ss>`.
   `silver_annotation/annotate_extra800.py` (stage `clips`, functions `parse` and `cut`)
   does exactly this with ffmpeg.
3. Keep the same relative paths under one folder, and set that folder as `video_base` in
   `config/pretrain_classification.py` (and `CLIP_ROOT` for `colab/run_table1.sh`).

## Clip lists (`clips/*.json`)

A JSON list with one entry per clip:

| Field | Meaning |
|---|---|
| `video` | relative path of the clip, also its ID everywhere else |
| `caption` | event class (one of the 13 below) |
| `comments_text` | the MatchTime commentary sentence for the event |
| `comments_text_anonymized` | the same sentence with players and teams replaced by `[PLAYER]` / `[TEAM]` |
| `league`, `match` | source competition and match |

Event classes (clips in the 1,000-clip set): goal 139, corner 137, yellow card 135,
foul with no card 135, lead to corner 133, substitution 131, injury 70, free kick 62,
penalty 25, foul lead to penalty 13, throw in 8, second yellow card 7, red card 5.

## Annotation files (`annotations/*.json`)

Standard COCO detection format, one *image* per annotated frame. The 30 frames of a clip
are sampled at 1 fps (the middle frame of each second), frames are 1280x720.

- `images[]`: `id` = `<video>_<second>`, `video_id` = the clip's `video`, `second` (0-29),
  `width`, `height`.
- `annotations[]`: `image_id`, `bbox` = `[x, y, w, h]` in pixels, `category_id`, `label`,
  and `no_roi`. A row with `no_roi: true` (`category_id: null`, `bbox: [0, 0, 0, 0]`)
  marks a frame in which the annotators found no cue, so it is a negative frame for the
  temporal metrics.
- In the merged file, images and annotations also carry `source`: `human` or
  `gemini-4-argon`.

**Cue tiers.** The three categories are the three cue tiers of the paper:

| `category_id` | `label` | Paper name |
|---|---|---|
| 1 | `small label` | Primary cue (P) |
| 2 | `large label` | Secondary cue (S) |
| 3 | `visual cue` | Common cue (C) |

The evaluation scores three nested label sets, P, P+S and P+S+C (`small_only`,
`small_large`, `small_large_visual_cues` in `inference/grounding_metrics.py`).

The human annotators boxed a subset of the 30 frames of each clip; the Gemini annotations
cover every frame. How the silver annotations were produced (prompts induced from the
human annotations, then applied to the 800 clips) is described in
`silver_annotation/README.md`.
