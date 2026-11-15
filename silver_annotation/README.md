# Silver annotations for the 800 extra clips

`annotations-coco-extra800.json` (repo root) was produced by Gemini 4 Argon in two stages.
No written cue definitions or per-event criteria are given to the model at any stage.

| Stage | Script | What it does |
|---|---|---|
| 1 | `induce_prompts_and_annotate_200.py` | For each event, shows the model up to 300 human-annotated frames of the 200 annotated clips (boxes drawn on the image and listed as text) and asks it to write annotation instructions. It then annotates the 2,711 human-annotated frames with those instructions and reports agreement with the human boxes. Output: `gemini-4-argon_nocrit_induced_prompts_folds1.json` (the 13 instruction sets). |
| 2 | `annotate_extra800.py` | Cuts the 800 clips from the SoccerNet 720p halves (15 s before to 15 s after each timestamp), samples 30 frames per clip (1 fps, middle frame of each second, as in the model evaluation) and annotates every frame with the stage 1 instructions for its event. Resumable. |
| 2b | `finish_extra800_colab.py` | Colab version of the annotation step: annotates only the frames still missing from the raw file and rebuilds the COCO files. |

Settings: model `gemini-4-argon`; thinking level `high` (stage 1) and `low` (stage 2); seed 0;
one frame per request in stage 2; boxes returned as `box_2d = [ymin, xmin, ymax, xmax]` on a
0-1000 grid and converted to pixels.

Result: 23,997 of 24,000 frames annotated (24,442 boxes). Three frames returned an answer that
could not be parsed on every attempt (the seed is fixed) and have no annotation; the spatial
metrics skip them and the temporal metrics count them as frames without a cue.

The scripts need a Gemini API key in `GOOGLE_API_KEY` (plus `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION=global` for a Vertex AI key) and, for stage 2, an rclone remote for the
storage that holds the SoccerNet halves and receives the clips (`RCLONE_REMOTE`, default `gdrive`).
The instruction sets written in stage 1 must be placed where stage 2 reads them
(`PROMPTS_REMOTE` in `annotate_extra800.py`).
