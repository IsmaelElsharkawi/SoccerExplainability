# Finishes the Gemini 4 Argon annotation of the 800 extra SoccerLens clips, on Colab.
#
# Picks up from the raw file on Drive (SoccerLens_LLM_annotations_gemini4_extra800/), annotates only the
# frames that are still missing or failed, with exactly the same settings as the RunPod run (same model,
# deduced prompts, frame sampling, thinking level, seed), then rebuilds the COCO files on Drive.
#
# In Colab (CPU runtime is enough):
#   from google.colab import drive; drive.mount('/content/drive')
#   import os
#   os.environ["GOOGLE_API_KEY"] = "..."            # the Gemini Enterprise key, as in your earlier notebook
#   os.environ["GOOGLE_CLOUD_PROJECT"] = "<YOUR_GCP_PROJECT>"
#   os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
#   !pip -q install -U "google-genai>=2.3.0"
#   !python -u silver_annotation/finish_extra800_colab.py
#
# Safe to rerun: every answer is saved as it arrives and copied to Drive every CHECKPOINT frames.
# Options (environment variables): N_WORKERS (default 16), CHECKPOINT (default 200), DRY_RUN=1 (no API calls).

import os, re, io, json, time, shutil, hashlib, random, threading, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

DRIVE      = "/content/drive/MyDrive"
OUT_DRIVE  = f"{DRIVE}/SoccerLens_LLM_annotations_gemini4_extra800"
RAW_DRIVE  = f"{OUT_DRIVE}/gemini-4-argon_extra800_raw.jsonl"
PROMPTS    = f"{OUT_DRIVE}/gemini-4-argon_nocrit_induced_prompts_folds1.json"
CLIP_LIST  = f"{DRIVE}/extra_clips.json"
CLIP_ROOT  = f"{DRIVE}/SoccerLens"
WORK       = "/content/gemini800"
RAW        = f"{WORK}/gemini-4-argon_extra800_raw.jsonl"
FRAMES     = f"{WORK}/frames"

# identical to extra800_pipeline.py (the RunPod run)
MODEL_ID      = "gemini-4-argon"
SECONDS       = list(range(30))
ANNOT_THINK   = "low"
ANNOT_MAX_OUT = 8192
GRID          = 1000
CAT_ID        = {"small label": 1, "large label": 2, "visual cue": 3}
PRICES        = [("2026-11-30", {"in": 2.0, "out": 10.0}), ("9999-12-31", {"in": 4.0, "out": 20.0})]
# a little more patient than the pod run, for a busy endpoint
N_WORKERS     = int(os.environ.get("N_WORKERS", 16))
MAX_RETRIES   = 8
TIMEOUT_S     = 300
CHECKPOINT    = int(os.environ.get("CHECKPOINT", 200))
DRY_RUN       = os.environ.get("DRY_RUN") == "1"

def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)

# ----------------------------- inputs -----------------------------
log("=== 1. checking inputs ===")
if not os.path.isdir(DRIVE):
    die("Drive is not mounted: run drive.mount('/content/drive') first")
for p in (RAW_DRIVE, PROMPTS, CLIP_LIST):
    if not os.path.isfile(p):
        die(f"missing {p}")
    log(f"found {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
os.makedirs(FRAMES, exist_ok=True)

clips = list({c["video"]: c for c in json.load(open(CLIP_LIST))}.values())
ev_of = {c["video"]: c["caption"] for c in clips}
log(f"clip list: {len(clips)} clips | " + ", ".join(f"{e} {n}" for e, n in Counter(ev_of.values()).most_common()))

induced = json.load(open(PROMPTS))
guides = {k.split("|")[0]: v["prompt"] for k, v in induced.items() if k.endswith("|0") and v.get("prompt")}
no_guide = sorted(set(ev_of.values()) - set(guides))
log(f"deduced prompts: {len(guides)} events" + (f" | no prompt for {no_guide} (base prompt used, as on the pod)" if no_guide else ""))

def build_prompt(event, guide):
    # identical to the 2,711-frame run and the RunPod run (no written criteria)
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

md5 = {e: hashlib.md5(build_prompt(e, guides.get(e)).encode()).hexdigest() for e in set(ev_of.values())}

def read_jsonl(path):
    rows, bad = [], 0
    if os.path.exists(path):
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad

# ----------------------------- what is left -----------------------------
log("")
log("=== 2. reading the existing annotations ===")
stamp = time.strftime("%Y%m%d_%H%M%S")
backup = f"{RAW_DRIVE}.before_colab_{stamp}"
shutil.copy(RAW_DRIVE, backup)
log(f"backup of the Drive raw file: {backup}")
rows, cut = read_jsonl(RAW_DRIVE)
reasons = Counter()
keep = {}
for r in rows:
    if r.get("video_id") not in ev_of:
        reasons["not in clip list"] += 1
    elif r.get("boxes") is None:
        reasons["error or unparseable (" + (r.get("raw") or "")[:40].replace("\n", " ") + "...)"] += 1
    elif r.get("prompt_md5") != md5[r["event"]]:
        reasons["made with a different prompt"] += 1
    else:
        keep[(r["video_id"], r["second"])] = r
log(f"raw file: {len(rows)} rows ({cut} cut-off lines) -> {len(keep)} valid frames kept")
for why, n in reasons.most_common():
    log(f"  dropped {n:5d}: {why}")
with open(RAW, "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in keep.values())

done_per_clip = Counter(v for v, _ in keep)
todo = [(c["video"], s) for c in clips for s in SECONDS if (c["video"], s) not in keep]
todo_clips = sorted({v for v, _ in todo})
status = Counter("complete" if done_per_clip[c["video"]] == 30 else "partial" if done_per_clip[c["video"]] else "not started"
                 for c in clips)
log(f"clips: {status['complete']} complete, {status['partial']} partial, {status['not started']} not started")
log(f"frames to annotate: {len(todo)} in {len(todo_clips)} clips")
for e, n in Counter(ev_of[v] for v, _ in todo).most_common():
    log(f"  {e:20s} {n:5d} frames")
if not todo:
    log("nothing left to annotate")

# ----------------------------- frames (same 'middle' sampling as the model evaluation) -----------------------------
import numpy as np, cv2
from PIL import Image

def middle_indices(vlen, num_frames=30):
    acc = min(num_frames, vlen)
    iv = np.linspace(0, vlen, acc + 1).astype(int)
    idx = [(iv[i] + iv[i + 1] - 1) // 2 for i in range(acc)]
    return idx + [idx[-1]] * (num_frames - len(idx))

def frame_dir(v):
    return f"{FRAMES}/{hashlib.md5(v.encode()).hexdigest()}"

def extract_frames(v):
    d = frame_dir(v)
    if all(os.path.exists(f"{d}/{k:02d}.jpg") for k in range(30)):
        return v, "cached"
    src = f"{CLIP_ROOT}/{v}"
    if not os.path.isfile(src):
        return v, f"MISSING clip {src}"
    local = f"{WORK}/tmp_{hashlib.md5(v.encode()).hexdigest()}.mp4"
    shutil.copy(src, local)                                   # Drive FUSE is slow to read frame by frame
    cap = cv2.VideoCapture(local); vlen = 0
    while cap.grab():
        vlen += 1
    cap.release()
    picks = middle_indices(vlen); need = set(picks); got = {}
    cap = cv2.VideoCapture(local)
    for fi in range(vlen):
        ok, f = cap.read()
        if not ok:
            break
        if fi in need:
            got[fi] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); os.remove(local)
    os.makedirs(d, exist_ok=True)
    for k, p in enumerate(picks):
        got[p].save(f"{d}/{k:02d}.jpg", format="JPEG", quality=95)
    return v, f"{vlen} video frames -> 30 sampled"

if todo:
    log("")
    log(f"=== 3. extracting frames for {len(todo_clips)} clips ===")
    missing = []
    with ThreadPoolExecutor(8) as ex:
        for i, (v, msg) in enumerate(ex.map(extract_frames, todo_clips), 1):
            log(f"  [{i}/{len(todo_clips)}] {v}: {msg}")
            if msg.startswith("MISSING"):
                missing.append(v)
    if missing:
        log(f"{len(missing)} clips are missing on Drive and are skipped")
        todo = [(v, s) for v, s in todo if v not in set(missing)]

# ----------------------------- Gemini -----------------------------
from google import genai
from google.genai import types

usage = defaultdict(float); lock = threading.Lock()

def rate():
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return next(r for last, r in PRICES if day <= last)

def make_client():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        die("GOOGLE_API_KEY is not set: set it with os.environ in a notebook cell before running this script")
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT")
    loc = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    http = types.HttpOptions(timeout=TIMEOUT_S * 1000)
    if key.startswith("AQ.") or os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1"):
        if not proj:
            die("GOOGLE_CLOUD_PROJECT is not set (needed for the Gemini Enterprise key)")
        log(f"client: Vertex AI, project {proj}, location {loc}, timeout {TIMEOUT_S} s")
        return genai.Client(vertexai=True, api_key=key, project=proj, location=loc, http_options=http)
    log(f"client: Gemini Developer API, timeout {TIMEOUT_S} s")
    return genai.Client(vertexai=False, api_key=key, http_options=http)

cfg = types.GenerateContentConfig(max_output_tokens=ANNOT_MAX_OUT, seed=0,
                                  thinking_config=types.ThinkingConfig(thinking_level=ANNOT_THINK))

def gemini(client, jpeg, prompt, tag):
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
            return r.text or "", attempt
        except Exception as ex:
            msg = str(ex)
            retryable = re.search(r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|time|Timeout|"
                                  r"ReadError|ConnectError|RemoteProtocol|disconnect", msg, re.I)
            if "Spend cap" in msg or attempt == MAX_RETRIES - 1 or not retryable:
                raise
            kind = "rate-limited (429)" if re.search(r"429|RESOURCE_EXHAUSTED", msg) else "retryable error"
            with lock:
                usage["throttled" if kind.startswith("rate") else "retries"] += 1
            wait = min(2 ** attempt * 5, 180) + random.random() * 3
            log(f"    {tag}: {kind}, attempt {attempt + 1}/{MAX_RETRIES}, waiting {wait:.0f} s | {msg[:120]}")
            time.sleep(wait)

if todo and not DRY_RUN:
    log("")
    log("=== 4. checking that Gemini 4 Argon answers ===")
    client = make_client()
    t = time.time()
    try:
        r = client.models.generate_content(model=MODEL_ID, contents="Reply with the word OK.",
                                           config=types.GenerateContentConfig(max_output_tokens=256))
        log(f"Argon answered in {time.time() - t:.1f} s: {(r.text or '').strip()[:30]!r}")
    except Exception as ex:
        log(f"Argon test request failed after {time.time() - t:.1f} s: {str(ex)[:300]}")
        if re.search(r"429|RESOURCE_EXHAUSTED", str(ex)):
            die("Argon is still rate-limited/blocked for this project (429). Nothing was annotated; try again later "
                "or ask for a quota increase (Cloud console > IAM & Admin > Quotas, filter gemini-4-argon).")
        if "Spend cap" in str(ex):
            die("the project's spend cap is reached; raise it and rerun")
        die("fix the error above and rerun")

# ----------------------------- annotate -----------------------------
if todo:
    log("")
    log(f"=== 5. annotating {len(todo)} frames with {MODEL_ID} ({N_WORKERS} parallel requests"
        + (", DRY RUN: no API calls" if DRY_RUN else "") + ") ===")
    st = {"n": 0, "ok": 0, "empty": 0, "bad": 0, "err": 0, "t0": time.time(), "since_ckpt": 0}
    fh = open(RAW, "a")

    def checkpoint(final=False):
        shutil.copy(RAW, RAW_DRIVE + ".tmp"); os.replace(RAW_DRIVE + ".tmp", RAW_DRIVE)
        log(f"  checkpoint: raw file copied to Drive ({sum(1 for _ in open(RAW))} rows)" + (" [final]" if final else ""))

    def summary():
        el = time.time() - st["t0"]; rate_m = st["n"] / max(el / 60, 1e-9)
        left = len(todo) - st["n"]
        log(f"  --- {st['n']}/{len(todo)} frames ({100 * st['n'] / len(todo):.1f}%) | {st['ok']} with boxes, "
            f"{st['empty']} empty, {st['bad']} unparseable, {st['err']} failed | {rate_m:.1f} frames/min | "
            f"ETA {left / max(rate_m, 1e-9):.0f} min | {usage['requests']:.0f} requests, {usage['throttled']:.0f} "
            f"rate-limited, {usage['retries']:.0f} other retries | ${usage['cost']:.2f} ---")

    def one(v, s):
        tag = f"{v} s{s:02d}"
        jpeg = open(f"{frame_dir(v)}/{s:02d}.jpg", "rb").read()
        W, H = Image.open(io.BytesIO(jpeg)).size
        e = ev_of[v]; t = time.time(); attempts = 0
        if DRY_RUN:
            txt = "[]"
        else:
            try:
                txt, attempts = gemini(client, jpeg, build_prompt(e, guides.get(e)), tag)
            except Exception as ex:
                txt = f"__ERROR__ {ex}"
        boxes = parse_boxes(txt)
        row = {"video_id": v, "second": s, "event": e, "prompt_md5": md5[e], "width": W, "height": H,
               "raw": txt, "boxes": boxes}
        with lock:
            if not DRY_RUN:
                fh.write(json.dumps(row) + "\n"); fh.flush()
            st["n"] += 1; st["since_ckpt"] += 1
            if txt.startswith("__ERROR__"):
                st["err"] += 1; what = f"FAILED: {txt[10:130]}"
            elif boxes is None:
                st["bad"] += 1; what = f"UNPARSEABLE answer: {txt[:80]!r}"
            elif not boxes:
                st["empty"] += 1; what = "no cue visible ([])"
            else:
                st["ok"] += 1
                what = f"{len(boxes)} boxes (" + ", ".join(f"{n} {l}" for l, n in Counter(b['label'] for b in boxes).items()) + ")"
            log(f"  [{st['n']}/{len(todo)}] {tag} [{e}]: {what} | {time.time() - t:.1f} s"
                + (f", {attempts} retries" if attempts else ""))
            if st["n"] % 50 == 0:
                summary()
            if st["since_ckpt"] >= CHECKPOINT and not DRY_RUN:
                fh.flush(); checkpoint(); st["since_ckpt"] = 0
            if st["err"] >= 50 and st["err"] > 0.5 * st["n"]:
                log("  more than half of the requests are failing; the remaining ones will fail too. "
                    "Stop the cell (finished frames are saved) and rerun later.")

    with ThreadPoolExecutor(N_WORKERS) as ex:
        for f in as_completed([ex.submit(one, v, s) for v, s in todo]):
            f.result()
    fh.close()
    summary()
    if not DRY_RUN:
        checkpoint(final=True)

# ----------------------------- COCO export -----------------------------
log("")
log("=== 6. rebuilding the COCO files ===")
rows, _ = read_jsonl(RAW)
best = {}
for r in rows:
    if r.get("boxes") is not None and r.get("prompt_md5") == md5.get(r.get("event")):
        best[(r["video_id"], r["second"])] = r
images, anns, aid = [], [], 1
for k in sorted(best):
    r = best[k]
    iid = f'{r["video_id"]}_{r["second"]}'; W, H = r["width"], r["height"]
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
        anns.append({"id": aid, "image_id": iid, "category_id": None, "bbox": [0, 0, 0, 0], "iscrowd": 0, "no_roi": True}); aid += 1
cats = [{"id": 1, "name": "small label"}, {"id": 2, "name": "large label"}, {"id": 3, "name": "visual cue"}]
info = {"description": f"SoccerLens extra clips, {MODEL_ID} annotations with deduced prompts (silver labels)",
        "prompts": os.path.basename(PROMPTS), "date": time.strftime("%Y-%m-%d")}
coco = {"info": info, "images": images, "annotations": anns, "categories": cats}

per_clip = Counter(v for v, _ in best)
full = {v for v, n in per_clip.items() if n == 30}
fi = [im for im in images if im["video_id"] in full]; fids = {im["id"] for im in fi}
full_coco = {"info": {**info, "description": info["description"] + f" | only the {len(full)} clips with all 30 frames annotated"},
             "images": fi, "annotations": [a for a in anns if a["image_id"] in fids], "categories": cats}

if DRY_RUN:
    log("DRY RUN: nothing written to Drive")
else:
    for name, obj in ((f"{MODEL_ID}_extra800_annotations-coco.json", coco),
                      (f"{MODEL_ID}_extra800_annotations-coco_full-clips.json", full_coco)):
        json.dump(obj, open(f"{WORK}/{name}", "w"))
        shutil.copy(f"{WORK}/{name}", f"{OUT_DRIVE}/{name}")
        log(f"wrote {OUT_DRIVE}/{name} ({len(obj['images'])} frames, {sum(not a['no_roi'] for a in obj['annotations'])} boxes)")

left = [c["video"] for c in clips if per_clip[c["video"]] < 30]
log("")
log("=== done ===")
log(f"frames annotated: {len(best)}/{len(clips) * 30} | clips complete: {len(full)}/{len(clips)} | "
    f"incomplete: {len(left)}")
for e, n in Counter(ev_of[v] for v in full).most_common():
    log(f"  {e:20s} {n:4d} complete clips")
if left:
    log(f"rerun this script to retry the {len(clips) * 30 - len(best)} missing frames, e.g. {left[:3]}")
log(f"this run: {usage['requests']:.0f} requests, {usage['in'] / 1e6:.2f}M input / {usage['out'] / 1e6:.2f}M output tokens, "
    f"about ${usage['cost']:.2f} at the promo price (about ${2 * usage['cost']:.2f} counted against the spend cap)")
