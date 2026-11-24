"""Build the static SoccerLens attribution explorer shipped with the supplementary material.

Usage:
    python demo/build_demo.py --raw <SoccerLens-for-annotation dir> --outputs <chefer output root> \
        [--soccermaster_dir <dir with SoccerMaster saliency .npz/.json>] [--per_class 4] [--out demo/site]

Writes <out>/index.html (standalone page, opens from file:// without a server), <out>/data.js and one
JPEG frame sheet per clip in <out>/frames/. Nothing is fetched at runtime besides an optional web font.

Inputs
  frames  <raw>/<league>/<match>/<clip>.mp4, sampled like the eval loader (30 frames, video frame 30*t+14)
  maps    <outputs>/output_chefer_{matchvision_temporal,siglip,soccermaster_temporal}/saliency/<clip>.{npz,json}
          npz: saliency_maps (30,h,w) in [0,1], frame_scores (30,), frame_{energy,pointing,s_iou,s_auc} (30,)
          json: prediction_text, ground_truth_text, summary (spatial + temporal metrics per cue tier)
  boxes   annotations-coco.json (human-annotated gold clips); small label = primary, large label = secondary,
          visual cue = common cue.
"""
import argparse, base64, glob, json, os, random, re, shutil
from collections import defaultdict
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--outputs", required=True)
ap.add_argument("--soccermaster_dir", default=None, help="override the SoccerMaster saliency dir")
ap.add_argument("--coco", default=os.path.join(REPO, "data/annotations/annotations-coco.json"))
ap.add_argument("--clips", default=os.path.join(REPO, "data/clips/selected_videos_for_annotations.json"))
ap.add_argument("--per_class", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frame_w", type=int, default=480)
ap.add_argument("--out", default=os.path.join(HERE, "site"))
args = ap.parse_args()

MODELS = [
    # key, label, saliency dir, note
    ("matchvision", "MatchVision", f"{args.outputs}/output_chefer_matchvision_temporal/saliency", "Chefer-T"),
    ("siglip", "SigLIP", f"{args.outputs}/output_chefer_siglip/saliency", "Chefer (no temporal attention)"),
    ("soccermaster", "SoccerMaster", args.soccermaster_dir or f"{args.outputs}/output_chefer_soccermaster_temporal/saliency", "Chefer-T"),
]
TIER = {"small label": "P", "large label": "S", "visual cue": "C"}
SUMMARY_TIERS = {"P": "small_only", "PS": "small_large", "PSC": "small_large_visual_cues"}
# Clips shown in the paper figures are always included (Fig. 2, Fig. 4 and the appendix figure).
PAPER = {
    "england_epl_2015-2016/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea/1_19_05.mp4": "Fig. 2",
    "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/2_32_19.mp4": "Fig. 2",
    "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/1_30_30.mp4": "Fig. 4",
    "italy_serie-a_2016-2017/2016-12-04 - 17-00 Lazio 0 - 2 AS Roma/2_20_48.mp4": "Fig. 4",
    "spain_laliga_2015-2016/2015-09-19 - 17-00 Real Madrid 1 - 0 Granada CF/2_24_39.mp4": "Fig. 4",
    "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/2_37_49.mp4": "Fig. 4",
    "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/2_21_33.mp4": "Fig. 4",
    "germany_bundesliga_2015-2016/2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen/2_27_40.mp4": "Fig. 4",
    "spain_laliga_2014-2015/2015-04-25 - 17-00 Espanyol 0 - 2 Barcelona/2_45_50.mp4": "Supp.",
    "england_epl_2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea/1_13_13.mp4": "Supp.",
    "england_epl_2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea/2_44_07.mp4": "Supp.",
    "europe_uefa-champions-league_2015-2016/2015-11-03 - 22-45 Sevilla 1 - 3 Manchester City/1_36_40.mp4": "Supp.",
    "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/2_33_10.mp4": "Supp.",
    "england_epl_2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool/1_41_54.mp4": "Supp.",
}


def index_model(d):
    """clip_id -> (npz path, json dict)"""
    out = {}
    for jf in glob.glob(f"{d}/*.json"):
        j = json.load(open(jf, encoding="utf-8"))
        npz = jf[:-5] + ".npz"
        if os.path.exists(npz):
            out[j["clip_id"]] = (npz, j)
    return out


def f32(x, nd=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def q8(a):
    return base64.b64encode(np.clip(np.round(a * 255), 0, 255).astype(np.uint8).tobytes()).decode()


sel = json.load(open(args.clips, encoding="utf-8"))
coco = json.load(open(args.coco, encoding="utf-8"))
idx = {m: index_model(d) for m, _, d, _ in MODELS}
for m in idx:
    print(f"{m}: {len(idx[m])} clips with maps")

boxes = defaultdict(lambda: defaultdict(list))
img_video = {im["id"]: (im["video_id"], im["second"], im["width"], im["height"]) for im in coco["images"]}
for a in coco["annotations"]:
    if a.get("no_roi") or a.get("label") not in TIER:
        continue
    vid, t, W, H = img_video[a["image_id"]]
    x, y, w, h = a["bbox"]
    boxes[vid][t].append([TIER[a["label"]], round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)])

usable = [s for s in sel if all(s["video"] in idx[m] for m in idx) and os.path.exists(f"{args.raw}/{s['video']}")]
print(f"{len(usable)} / {len(sel)} gold clips have maps from all models and a raw video")
by_class = defaultdict(list)
for s in usable:
    by_class[s["caption"]].append(s)
rng = random.Random(args.seed)
chosen = []
for cls in sorted(by_class):
    pool = sorted(by_class[cls], key=lambda s: s["video"])
    pinned = [s for s in pool if s["video"] in PAPER]
    rest = [s for s in pool if s["video"] not in PAPER]
    rng.shuffle(rest)
    chosen += pinned + rest[:max(0, args.per_class - len(pinned))]
print(f"selected {len(chosen)} clips:", {c: sum(s['caption'] == c for s in chosen) for c in by_class})

if os.path.isdir(args.out):
    shutil.rmtree(args.out)
os.makedirs(f"{args.out}/frames")

COLS, ROWS = 6, 5
fw = args.frame_w
fh = round(fw * 9 / 16)
clips = []
for i, s in enumerate(chosen):
    vid = s["video"]
    cap = cv2.VideoCapture(f"{args.raw}/{vid}")
    sheet = np.zeros((ROWS * fh, COLS * fw, 3), np.uint8)
    for t in range(30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30 * t + 14)
        ok, fr = cap.read()
        assert ok, (vid, t)
        r, c = divmod(t, COLS)
        sheet[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_AREA)
    cid = f"c{i:03d}"
    cv2.imwrite(f"{args.out}/frames/{cid}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1])

    league, match, stem = vid.split("/")
    date = match[:10]
    teams = re.sub(r"^\d{4}-\d{2}-\d{2} - \d{2}-\d{2} ", "", match)
    entry = {
        "id": cid, "label": s["caption"], "league": league, "match": f"{teams} ({date})",
        "time": f"half {stem[0]}, {stem[2:4]}:{stem[5:7]}", "paper": PAPER.get(vid),
        "boxes": {str(t): b for t, b in sorted(boxes[vid].items())}, "models": {},
    }
    for m, _, _, _ in MODELS:
        npz, j = idx[m][vid]
        z = np.load(npz, allow_pickle=False)
        sm = z["saliency_maps"].astype(np.float32)
        summ = j["summary"]
        tiers = {}
        for tk, name in SUMMARY_TIERS.items():
            g = summ["label_group_scores"].get(name, {})
            tl = summ["temporal_localization"]["tiers"].get(name, {})
            tiers[tk] = {
                "energy": f32(g.get("mean_energy_inside_bbox", np.nan)), "pointing": f32(g.get("mean_pointing_accuracy", np.nan)),
                "s_iou": f32(g.get("mean_iou", np.nan)), "s_auc": f32(g.get("mean_s_auc", np.nan)),
                "t_iou": f32(tl.get("tIoU", np.nan)), "t_auc": f32(tl.get("tAUC", np.nan)), "t_ap": f32(tl.get("tAP", np.nan)),
            }
        entry["models"][m] = {
            "pred": j["prediction_text"], "n": int(sm.shape[1]), "maps": q8(sm),
            "scores": [f32(v, 5) for v in z["frame_scores"]],
            "frame": {k: [f32(v) for v in z[f"frame_{k}"]] for k in ("energy", "pointing", "s_iou", "s_auc")},
            "tiers": tiers,
        }
    clips.append(entry)
    print(f"[{i + 1}/{len(chosen)}] {cid} {s['caption']:<22} {os.path.getsize(f'{args.out}/frames/{cid}.jpg') // 1024} KB")

data = {
    "frame": {"w": fw, "h": fh, "cols": COLS, "rows": ROWS, "n": 30},
    "models": [{"key": m, "name": n, "method": note} for m, n, _, note in MODELS],
    "clips": clips,
}
with open(f"{args.out}/data.js", "w", encoding="utf-8") as f:
    f.write("window.SOCCERLENS = " + json.dumps(data, separators=(",", ":")) + ";\n")

body = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
with open(f"{args.out}/index.html", "w", encoding="utf-8") as f:
    f.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
tot = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(args.out) for fn in fns)
print(f"wrote {args.out}  ({tot / 1e6:.1f} MB, data.js {os.path.getsize(args.out + '/data.js') / 1e6:.1f} MB)")
