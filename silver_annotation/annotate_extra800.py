# SoccerLens extra 800 clips, end to end on a RunPod machine:
#   stage "clips":    copy the SoccerNet 720p match halves the clips come from, cut each clip from 15 s before to 15 s
#                     after its timestamp, upload it to Google Drive under SoccerLens/<league>/<match>/<file>.mp4
#   stage "annotate": annotate every clip with Gemini 4 Argon using the per-event prompts it already deduced
#                     (gemini-4-argon_nocrit_induced_prompts_folds1.json), and save the boxes in COCO format.
#
# RunPod setup (network volume on /workspace, which keeps everything across pod restarts):
#   apt-get update && apt-get install -y ffmpeg && curl https://rclone.org/install.sh | bash
#   rclone config            # Google Drive remote named "gdrive" (or set RCLONE_REMOTE to your remote name)
#   python -m venv /workspace/venv && source /workspace/venv/bin/activate
#   pip install -U "google-genai>=2.3.0" opencv-python-headless numpy
#   export GOOGLE_API_KEY="AQ...."                 # the Gemini Enterprise key
#   export GOOGLE_CLOUD_PROJECT=<YOUR_GCP_PROJECT> GOOGLE_CLOUD_LOCATION=global GOOGLE_GENAI_USE_ENTERPRISE=true
#   python -u extra800_pipeline.py --clips /workspace/extra_clips.json            # both stages
#   python -u extra800_pipeline.py --clips /workspace/extra_clips.json --stage clips
#   python -u extra800_pipeline.py --clips /workspace/extra_clips.json --stage annotate --max-clips 3   # smoke test
#
# Everything is resumable: rerun the same command after a crash or disconnect and finished work is skipped.

import os, re, io, json, time, shutil, hashlib, random, threading, subprocess, argparse, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="/workspace/extra_clips.json", help="clip list JSON (video, caption, ...)")
ap.add_argument("--stage", default="both", choices=["clips", "annotate", "both"])
ap.add_argument("--max-clips", type=int, default=None, help="only the first N clips (smoke test)")
args = ap.parse_args()

# ----------------------------- config -----------------------------
WS            = os.environ.get("WORKSPACE", "/workspace")
REMOTE        = os.environ.get("RCLONE_REMOTE", "gdrive")       # name of the rclone Google Drive remote
SRC_REMOTE    = os.environ.get("SOCCERNET_REMOTE", f"{REMOTE}:SoccerNet_720p")  # SoccerNet_720p
DEST_REMOTE   = os.environ.get("SOCCERLENS_REMOTE", f"{REMOTE}:SoccerLens")
PROMPTS_REMOTE = f"{REMOTE}:SoccerLens_LLM_annotations_gemini4_induced/gemini-4-argon_nocrit_induced_prompts_folds1.json"
OUT_REMOTE    = os.environ.get("OUT_REMOTE", f"{REMOTE}:SoccerLens_LLM_annotations_gemini4_extra800")
SRC_DIR       = f"{WS}/SoccerNet_720p"         # downloaded match halves
CLIP_DIR      = f"{WS}/SoccerLens_extra"       # cut clips, same layout as on Drive
FRAME_DIR     = f"{WS}/soccerlens_extra_frames"
OUT_DIR       = f"{WS}/gemini4_extra800"
KEEP_SOURCES  = False          # True keeps every downloaded .mkv (about 1.1 GB per half, 98 halves)
BEFORE, AFTER = 15, 15         # clip = [timestamp - 15 s, timestamp + 15 s]
CRF, PRESET   = 18, "veryfast"
HALF_WORKERS  = max(2, min(6, (os.cpu_count() or 4) // 2))   # match halves at once (download + cut + upload)

MODEL_ID      = "gemini-4-argon"
SECONDS       = list(range(30))   # frames annotated per clip: all 30 (second k -> frame k); e.g. range(0, 30, 2) halves cost
ANNOT_THINK   = "low"          # same as the 2,711-frame run
ANNOT_MAX_OUT = 8192
N_WORKERS     = int(os.environ.get("N_WORKERS", 16))
MAX_RETRIES   = 6
TIMEOUT_S     = int(os.environ.get("TIMEOUT_S", 180))
PRICES        = [("2026-11-30", {"in": 2.0, "out": 10.0}), ("9999-12-31", {"in": 4.0, "out": 20.0})]
GRID          = 1000
LABELS        = ["small label", "large label", "visual cue"]
CAT_ID        = {"small label": 1, "large label": 2, "visual cue": 3}

for d in (SRC_DIR, CLIP_DIR, FRAME_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} ... failed: {(r.stderr or r.stdout).strip()[-600:]}")
    return r

for tool in ("ffmpeg", "ffprobe", "rclone"):
    if not shutil.which(tool):
        raise SystemExit(f"{tool} is missing: apt-get install -y ffmpeg && curl https://rclone.org/install.sh | bash")

clips = json.load(open(args.clips))
clips = list({c["video"]: c for c in clips}.values())
if args.max_clips:
    clips = clips[:args.max_clips]
ev = defaultdict(int)
for c in clips:
    ev[c["caption"]] += 1
print(f"{len(clips)} clips | " + ", ".join(f"{e} {n}" for e, n in sorted(ev.items(), key=lambda x: -x[1])))

def parse(video_id):
    """'league_season/match/1_13_09.mp4' -> (source half path relative to SoccerNet_720p, timestamp in seconds)"""
    league_season, match, fname = video_id.split("/")
    league, season = league_season.rsplit("_", 1)
    half, mm, ss = map(int, re.search(r"(\d)_(\d+)_(\d+)\.mp4$", fname).groups())
    return f"{league}/{season}/{match}/{half}_720p.mkv", mm * 60 + ss

def remote_join(remote, rel):
    return remote + ("" if remote.endswith((":", "/")) else "/") + rel

# ----------------------------- frames (same 'middle' sampling as the model evaluation) -----------------------------
import numpy as np, cv2
from PIL import Image

def middle_indices(vlen, num_frames=30):
    acc = min(num_frames, vlen)
    iv = np.linspace(0, vlen, acc + 1).astype(int)
    idx = [(iv[i] + iv[i + 1] - 1) // 2 for i in range(acc)]
    return idx + [idx[-1]] * (num_frames - len(idx))

def frame_dir(video_id):
    return f"{FRAME_DIR}/{hashlib.md5(video_id.encode()).hexdigest()}"

def frames_cached(video_id):
    return all(os.path.exists(f"{frame_dir(video_id)}/{k:02d}.jpg") for k in range(30))

def extract_frames(video_id):
    """30 JPEGs per clip: second k -> frame k of the clip's 'middle' sampling."""
    path = f"{CLIP_DIR}/{video_id}"
    cap = cv2.VideoCapture(path); vlen = 0
    while cap.grab():
        vlen += 1
    cap.release()
    picks = middle_indices(vlen); need = set(picks); got = {}
    cap = cv2.VideoCapture(path)
    for fi in range(vlen):
        ok, f = cap.read()
        if not ok:
            break
        if fi in need:
            got[fi] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    d = frame_dir(video_id); os.makedirs(d, exist_ok=True)
    for k, p in enumerate(picks):
        got[p].save(f"{d}/{k:02d}.jpg", format="JPEG", quality=95)

# ----------------------------- stage 1: clips -----------------------------
def existing_on_drive():
    r = sh(["rclone", "lsf", "-R", "--files-only", DEST_REMOTE], check=False)
    if r.returncode != 0:
        raise SystemExit(f"cannot list {DEST_REMOTE}: {r.stderr.strip()[-400:]}\nCheck `rclone lsd {REMOTE}:` works.")
    return set(r.stdout.split("\n")) - {""}

def reference_format(on_drive):
    """Resolution / frame rate / audio of a clip already in SoccerLens, so the new clips look the same."""
    ref = next((f for f in sorted(on_drive) if re.search(r"/\d_\d+_\d+\.mp4$", f)), None)
    if ref is None:
        return None
    local = f"{WS}/_ref.mp4"
    sh(["rclone", "copyto", remote_join(DEST_REMOTE, ref), local])
    out = json.loads(sh(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", local]).stdout)
    os.remove(local)
    v = next(s for s in out["streams"] if s["codec_type"] == "video")
    return {"ref": ref, "w": v["width"], "h": v["height"], "fps": v["r_frame_rate"], "pix_fmt": v.get("pix_fmt") or "yuv420p",
            "audio": any(s["codec_type"] == "audio" for s in out["streams"]), "dur": float(out["format"]["duration"])}

def cut(src, video_id, fmt, threads):
    _, t = parse(video_id)
    out = f"{CLIP_DIR}/{video_id}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t - BEFORE, 0)}", "-i", src, "-t", f"{BEFORE + AFTER}",
           "-map", "0:v:0", "-map", "0:a:0?",            # first video + first audio only (some halves carry extra streams)
           "-max_muxing_queue_size", "9999"]            # avoids "Too many packets buffered for output stream"
    if fmt:
        cmd += ["-vf", f"scale={fmt['w']}:{fmt['h']}", "-r", fmt["fps"], "-pix_fmt", fmt["pix_fmt"]]
    cmd += ["-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-threads", str(threads), "-movflags", "+faststart"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if (fmt is None or fmt["audio"]) else ["-an"]
    sh(cmd + [out + ".tmp.mp4"])
    os.replace(out + ".tmp.mp4", out)

def run_clips():
    print("\n=== stage 1: clips ===")
    on_drive = existing_on_drive()
    fmt = reference_format(on_drive)
    if fmt:
        print(f"matching existing SoccerLens clips ({fmt['ref']}): {fmt['w']}x{fmt['h']}, {fmt['fps']} fps, "
              f"audio {fmt['audio']}, {fmt['dur']:.1f} s")
    vids = [c["video"] for c in clips]
    to_cut = [v for v in vids if v not in on_drive and not local_ok(v)]
    to_upload = [v for v in vids if v not in on_drive and local_ok(v)]
    to_fetch = [v for v in vids if v in on_drive and not local_ok(v)]       # needed locally for the annotation stage
    print(f"{len(vids)} clips | {len(vids) - len(to_cut) - len(to_upload)} already on Drive | {len(to_upload)} cut but "
          f"not uploaded | {len(to_cut)} to cut")
    by_half = defaultdict(list)
    for v in to_cut:
        by_half[parse(v)[0]].append(v)
    threads = max((os.cpu_count() or 4) // HALF_WORKERS, 1)
    t0 = time.time(); lock = threading.Lock(); done_halves = [0]; failed = []

    def do_half(item):
        rel, vs = item
        src = f"{SRC_DIR}/{rel}"
        if not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            r = sh(["rclone", "copyto", remote_join(SRC_REMOTE, rel), src + ".part", "--retries", "5",
                    "--low-level-retries", "20"], check=False)
            if r.returncode != 0:
                return rel, vs, f"download failed: {r.stderr.strip()[-300:]}"
            os.replace(src + ".part", src)
        errs = []
        for v in vs:
            try:
                cut(src, v, fmt, threads)
            except Exception as ex:
                errs.append(f"{v}: {ex}")
        ok = [v for v in vs if local_ok(v)]
        if ok:                                                               # upload this half's clips right away
            lst = f"{WS}/_upload_{hashlib.md5(rel.encode()).hexdigest()}.txt"
            open(lst, "w").write("\n".join(ok) + "\n")
            r = sh(["rclone", "copy", CLIP_DIR, DEST_REMOTE, "--files-from", lst, "--transfers", "8"], check=False)
            os.remove(lst)
            if r.returncode != 0:
                errs.append(f"upload failed: {r.stderr.strip()[-300:]}")
        if not KEEP_SOURCES and os.path.exists(src):
            os.remove(src)
        return rel, vs, "; ".join(errs)

    with ThreadPoolExecutor(HALF_WORKERS) as ex:
        for fut in as_completed([ex.submit(do_half, it) for it in sorted(by_half.items())]):
            rel, vs, err = fut.result()
            with lock:
                done_halves[0] += 1
                if err:
                    failed.append((rel, err))
                print(f"  half {done_halves[0]}/{len(by_half)} | {len(vs)} clips | {(time.time() - t0) / 60:.1f} min | {rel}"
                      + (f" | ERROR {err}" if err else ""), flush=True)
    if to_upload:
        lst = f"{WS}/_upload_rest.txt"; open(lst, "w").write("\n".join(to_upload) + "\n")
        sh(["rclone", "copy", CLIP_DIR, DEST_REMOTE, "--files-from", lst, "--transfers", "8"]); os.remove(lst)
    for v in to_fetch:
        sh(["rclone", "copyto", remote_join(DEST_REMOTE, v), f"{CLIP_DIR}/{v}"])
    on_drive = existing_on_drive()
    missing = [v for v in vids if v not in on_drive]
    json.dump({"clips": len(vids), "missing_on_drive": missing, "failed_halves": failed, "format": fmt,
               "window": [-BEFORE, AFTER]}, open(f"{OUT_DIR}/clips_report.json", "w"), indent=1)
    print(f"clips on Drive: {len(vids) - len(missing)}/{len(vids)}" + (f" | missing {len(missing)}, see "
          f"{OUT_DIR}/clips_report.json" if missing else ""))

# ----------------------------- stage 2: Gemini annotation -----------------------------
def build_prompt(event, guide):
    # identical to the 2,711-frame run (annotate_gemini4_induce.py, no written criteria)
    p = (f'This frame is from a 30-second soccer broadcast clip of the event: "{event}".\n'
         'Annotate the visual cues for this event with bounding boxes in three categories: '
         '"small label", "large label" and "visual cue".\n\n')
    p += (f'Annotation instructions for "{event}" (follow them exactly):\n{guide}\n\n' if guide else
          "Box only cues that are visible in THIS frame. Use as many boxes per category as needed.\n")
    p += ('Return ONLY a JSON array, one object per box: [{"label": "<small label|large label|visual cue>", '
          f'"box_2d": [ymin, xmin, ymax, xmax]}}], coordinates normalized to 0-{GRID} relative to the image. '
          "If no cue for this event is visible in the frame, return [].")
    return p

def parse_boxes(text):
    """Gemini answer -> [{"label", "bbox_2d": [x1, y1, x2, y2]}] on the 0-1000 grid, or None if unparseable."""
    text = re.sub(r"```(?:json)?", "", text or "")
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for o in arr if isinstance(arr, list) else []:
        if isinstance(o, dict) and o.get("label") in CAT_ID and isinstance(o.get("box_2d"), (list, tuple)) and len(o["box_2d"]) == 4:
            try:
                ymin, xmin, ymax, xmax = [float(v) for v in o["box_2d"]]
            except (TypeError, ValueError):
                continue
            out.append({"label": o["label"], "bbox_2d": [xmin, ymin, xmax, ymax]})
    return out

def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return rows

def local_ok(v):
    p = f"{CLIP_DIR}/{v}"
    return os.path.exists(p) and os.path.getsize(p) > 100_000

def run_annotate(clip_thread=None):
    """Annotate clips as they become available. With clip_thread running, clips are picked up as soon as they are cut,
    so annotation overlaps with downloading and cutting instead of waiting for all 800."""
    print("\n=== stage 2: Gemini annotation" + (" (runs alongside stage 1)" if clip_thread else "") + " ===")
    vids_all = [c["video"] for c in clips]
    ev_of = {c["video"]: c["caption"] for c in clips}

    prompts_local = f"{OUT_DIR}/{os.path.basename(PROMPTS_REMOTE)}"
    if not os.path.exists(prompts_local):
        sh(["rclone", "copyto", PROMPTS_REMOTE, prompts_local])
    induced = json.load(open(prompts_local))
    guides = {k.split("|")[0]: v["prompt"] for k, v in induced.items() if k.endswith("|0") and v.get("prompt")}
    no_guide = sorted(set(ev_of.values()) - set(guides))
    print(f"{len(guides)} deduced prompts loaded" + (f" | no prompt for {no_guide} (base prompt used)" if no_guide else ""))

    from google import genai
    from google.genai import types
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GOOGLE_API_KEY (the Gemini Enterprise key) before the annotation stage")
    http = types.HttpOptions(timeout=TIMEOUT_S * 1000)
    if os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1") or (key or "").startswith("AQ."):
        proj = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not proj:
            raise SystemExit("set GOOGLE_CLOUD_PROJECT (and GOOGLE_CLOUD_LOCATION=global) for the Gemini Enterprise key")
        client = genai.Client(vertexai=True, api_key=key, project=proj,
                              location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global", http_options=http)
    else:
        client = genai.Client(vertexai=False, api_key=key, http_options=http)
    cfg = types.GenerateContentConfig(max_output_tokens=ANNOT_MAX_OUT, seed=0,
                                      thinking_config=types.ThinkingConfig(thinking_level=ANNOT_THINK))
    usage = defaultdict(float); lock = threading.Lock()

    def rate():
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return next(r for last, r in PRICES if day <= last)

    def gemini(jpeg, prompt):
        if os.environ.get("MOCK_GEMINI"):                               # offline test
            time.sleep(0.05)
            return '[{"label": "small label", "box_2d": [100, 200, 300, 400]}]'
        for attempt in range(MAX_RETRIES):
            try:
                r = client.models.generate_content(model=MODEL_ID, config=cfg, contents=[
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), types.Part.from_text(text=prompt)])
                u = r.usage_metadata; rt = rate()
                i = getattr(u, "prompt_token_count", 0) or 0
                o = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
                with lock:
                    usage["in"] += i; usage["out"] += o; usage["requests"] += 1
                    usage["cost"] += i / 1e6 * rt["in"] + o / 1e6 * rt["out"]
                return r.text or ""
            except Exception as ex:
                msg = str(ex)
                if attempt == MAX_RETRIES - 1 or not re.search(
                        r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|time|Timeout|ReadError|ConnectError|RemoteProtocol",
                        msg, re.I):
                    raise
                with lock:
                    usage["throttled" if re.search(r"429|RESOURCE_EXHAUSTED", msg) else "retries"] += 1
                time.sleep(min(2 ** attempt * 5, 120) + random.random() * 3)

    pred_jsonl = f"{OUT_DIR}/{MODEL_ID}_extra800_raw.jsonl"
    md5 = lambda e: hashlib.md5(build_prompt(e, guides.get(e)).encode()).hexdigest()
    rows = [r for r in read_jsonl(pred_jsonl) if r.get("boxes") is not None and r.get("prompt_md5") == md5(r["event"])]
    with open(pred_jsonl, "w") as fh:                                   # rewrite: drops cut-off / unparseable / stale rows
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    done = {(r["video_id"], r["second"]) for r in rows}
    print(f"{len(vids_all)} clips x {len(SECONDS)} frames | {len(done)} frames already done | {N_WORKERS} parallel requests")

    fh = open(pred_jsonl, "a")
    st = {"n": 0, "bad": 0, "submitted": 0, "t0": time.time()}

    def progress(force=False):
        if force or st["n"] % 200 == 0:
            print(f"  {st['n']} frames annotated | {st['submitted'] - st['n']} queued | {st['bad']} unparseable | "
                  f"{(time.time() - st['t0']) / 60:.1f} min | {usage['requests']:.0f} requests, "
                  f"{usage['throttled']:.0f} rate-limited, {usage['retries']:.0f} other retries | ${usage['cost']:.2f}", flush=True)

    def one(v, s):
        jpeg = open(f"{frame_dir(v)}/{s:02d}.jpg", "rb").read()
        W, H = Image.open(io.BytesIO(jpeg)).size
        e = ev_of[v]
        try:
            txt = gemini(jpeg, build_prompt(e, guides.get(e)))
        except Exception as ex:
            txt = f"__ERROR__ {ex}"
        row = {"video_id": v, "second": s, "event": e, "prompt_md5": md5(e), "width": W, "height": H,
               "raw": txt, "boxes": parse_boxes(txt)}
        with lock:
            fh.write(json.dumps(row) + "\n"); fh.flush()
            st["n"] += 1; st["bad"] += row["boxes"] is None
            progress()
            if st["n"] % 2000 == 0:                                     # checkpoint to Drive
                sh(["rclone", "copy", OUT_DIR, OUT_REMOTE, "--include", "*.jsonl", "--include", "*.json"], check=False)

    api_pool = ThreadPoolExecutor(N_WORKERS)
    frame_pool = ThreadPoolExecutor(os.cpu_count() or 4)
    scheduled, skipped = set(), []

    def schedule(vs):
        vs = [v for v in vs if v not in scheduled]
        need = [v for v in vs if not frames_cached(v)]
        if need:
            list(frame_pool.map(extract_frames, need))
        for v in vs:
            scheduled.add(v)
            for s in SECONDS:
                if (v, s) not in done:
                    api_pool.submit(one, v, s); st["submitted"] += 1

    while clip_thread is not None and clip_thread.is_alive():            # stream clips while stage 1 is still cutting
        schedule([v for v in vids_all if v not in scheduled and local_ok(v)])
        time.sleep(20)
    final = []
    for v in vids_all:
        if v in scheduled:
            continue
        if not local_ok(v):                                             # cut on another machine or earlier: fetch it
            sh(["rclone", "copyto", remote_join(DEST_REMOTE, v), f"{CLIP_DIR}/{v}"], check=False)
        (final if local_ok(v) else skipped).append(v)
    schedule(final)
    if skipped:
        print(f"  {len(skipped)} clips not available locally or on Drive, not annotated (see clips_report.json)")
    api_pool.shutdown(wait=True); frame_pool.shutdown(wait=True); fh.close()
    progress(force=True)

    # COCO export, same structure as annotations-coco.json (pixel boxes, empty frames as no_roi entries)
    rows = read_jsonl(pred_jsonl)
    images, anns, aid = [], [], 1
    for r in sorted(rows, key=lambda r: (r["video_id"], r["second"])):
        if r["boxes"] is None:
            continue
        iid = f'{r["video_id"]}_{r["second"]}'
        W, H = r["width"], r["height"]
        images.append({"id": iid, "file_name": f'{r["video_id"]}_t{r["second"]}.jpg', "video_id": r["video_id"],
                       "second": r["second"], "width": W, "height": H, "event": r["event"]})
        boxes = []
        for b in r["boxes"]:
            x1, y1, x2, y2 = b["bbox_2d"]
            x1, x2 = sorted((min(max(x1, 0), GRID), min(max(x2, 0), GRID)))
            y1, y2 = sorted((min(max(y1, 0), GRID), min(max(y2, 0), GRID)))
            if x2 > x1 and y2 > y1:
                boxes.append((b["label"], [x1 / GRID * W, y1 / GRID * H, (x2 - x1) / GRID * W, (y2 - y1) / GRID * H]))
        for lab, bb in boxes:
            anns.append({"id": aid, "image_id": iid, "category_id": CAT_ID[lab], "bbox": bb, "iscrowd": 0,
                         "no_roi": False, "label": lab}); aid += 1
        if not boxes:
            anns.append({"id": aid, "image_id": iid, "category_id": None, "bbox": [0, 0, 0, 0], "iscrowd": 0,
                         "no_roi": True}); aid += 1
    coco = {"info": {"description": f"SoccerLens extra clips, {MODEL_ID} annotations with deduced prompts (silver labels)",
                     "prompts": os.path.basename(PROMPTS_REMOTE), "date": time.strftime("%Y-%m-%d")},
            "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "small label"}, {"id": 2, "name": "large label"}, {"id": 3, "name": "visual cue"}]}
    coco_path = f"{OUT_DIR}/{MODEL_ID}_extra800_annotations-coco.json"
    json.dump(coco, open(coco_path, "w"))
    n_bad = sum(r["boxes"] is None for r in rows)
    print(f"\n{len(images)} frames in COCO | {sum(not a['no_roi'] for a in anns)} boxes | "
          f"{sum(a['no_roi'] for a in anns)} empty frames | {n_bad} unparseable (rerun to retry them)")
    print(f"this run: {usage['requests']:.0f} requests, {usage['in'] / 1e6:.2f}M input, {usage['out'] / 1e6:.2f}M output "
          f"tokens, about ${usage['cost']:.2f} at list price (Cloud billing has the real figure)")
    sh(["rclone", "copy", OUT_DIR, OUT_REMOTE, "--include", "*.jsonl", "--include", "*.json"])
    print(f"saved to {OUT_REMOTE}")

def _clips_safe():
    try:
        run_clips()
    except BaseException as ex:                                         # report, and let annotation finish what it has
        import traceback; traceback.print_exc()
        print(f"stage 1 stopped: {ex}")

if args.stage == "clips":
    run_clips()
elif args.stage == "annotate":
    run_annotate()
else:                                                                   # both: annotate clips as soon as they are cut
    t = threading.Thread(target=_clips_safe, daemon=True)
    t.start()
    run_annotate(clip_thread=t)
    t.join()
