# SoccerLens: let Gemini 4 Argon deduce the annotation prompt for each event from the human-labelled frames,
# then annotate all 200 clips with the deduced prompts. Same exercise as annotate_qwen38_induce.py, run from Colab
# against the Gemini API (no GPU needed).
#
# Colab, one cell before running:
#   from google.colab import drive, userdata; drive.mount("/content/drive")
#   import os; os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")   # key stored in Colab secrets
#   !pip -q install -U "google-genai>=2.3.0" opencv-python-headless
#   !apt-get -qq install -y ffmpeg
#   !python -u annotate_gemini4_induce.py
#
# Frames: the script uses the same 'middle' sampling and the same JPEG cache layout as the Qwen script, so both models
# see identical frames. It looks for them in this order:
#   1. /content/soccerlens_frames (already unpacked in this runtime)
#   2. MyDrive/soccerlens_frames.tar (packed by an earlier run, or by the Qwen script with SOCCERLENS_MODE=frames;
#      from the RunPod pod:  tar -cf /workspace/soccerlens_frames.tar -C /workspace/soccerlens_frames .
#                            rclone copy /workspace/soccerlens_frames.tar gdrive: )
#   3. cut from MyDrive/SoccerNet_720p (add a shortcut to the shared SoccerNet_720p folder in My Drive), then packed to
#      MyDrive/soccerlens_frames.tar so a runtime reset does not cut them again.
#
# Differences from the Qwen script, all because of the model:
#   - Boxes are exchanged in Gemini's native format, box_2d = [ymin, xmin, ymax, xmax] on a 0-1000 grid, and converted
#     to [x1, y1, x2, y2] before saving, so the saved rows and the scoring are identical to the Qwen run.
#   - Gemini has a 1M token context, so stage 1 can show many more example frames per event (INDUCE_MAX_IMAGES).
#   - Thinking is hidden from the answer text, so the analysis never truncates the answer; a failed answer is retried
#     instead of finished in a follow-up request.
#   - Requests run in parallel threads with retry and backoff; every result is appended to the JSONL as it arrives.
#
# INDUCE_FOLDS: 1 = prompt deduced from all clips of the event (upper bound, as in the Qwen run);
#               2 = cross-fitted, each clip annotated with a prompt deduced from the other half (held-out).

import os, re, io, json, base64, subprocess, sys, time, random, hashlib, shutil, tarfile, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- config -----------------------------
MODEL_ID        = os.environ.get("SOCCERLENS_GEMINI_MODEL", "gemini-4-argon")
API             = "auto"            # "auto" = try the Interactions API, fall back to generate_content | or force one
MODE            = os.environ.get("SOCCERLENS_MODE", "both")   # "induce" | "annotate" | "both" | "frames"
INDUCE_FOLDS    = 1
REINDUCE        = False
# Whether Gemini sees the paper's written cue definitions and per-event criteria (TIER_DEF, CUE_SCHEMA).
USE_CRITERIA_IN_INDUCTION  = False   # False = prompts are deduced from the human boxes alone
USE_CRITERIA_IN_ANNOTATION = False   # False = stage 2 gets only the deduced instructions (no written criteria either)
SEED            = 0
INDUCE_THINK    = "high"            # thinking level for prompt induction: minimal | low | medium | high
ANNOT_THINK     = "low"             # thinking level for annotation (2,711 requests): low keeps cost and latency down
INDUCE_MAX_IMAGES = 300             # example frames per induction request, spread evenly over the source clips
INDUCE_MAX_SIDE = 768               # example frames are downscaled to this width (boxes drawn on them)
INDUCE_RES      = "medium"          # per-image resolution for examples (Interactions API): low|medium|high|ultra_high|None
ANNOT_RES       = "high"            # per-image resolution for the frame being annotated (small balls need detail)
INDUCE_MAX_WORDS = 300
INDUCE_MAX_OUT  = 32768             # output token limit for induction (thinking counts against it)
ANNOT_MAX_OUT   = 8192
SECONDS_MODE    = "annotated"       # "annotated" = the 2,711 COCO frames, as in the Qwen run; "all" = 30 per clip
N_WORKERS       = 16                # parallel requests in stage 2; lower it if you hit 429s
MAX_RETRIES     = 6
REQUEST_TIMEOUT_S = 180             # a request that takes longer is abandoned and retried
MAX_CLIPS       = None              # smoke test, e.g. 5; None = all 200 (resume skips done frames)
ONLY_EVENTS     = None              # e.g. ["corner"]
# Gemini 4 Argon list prices, USD per 1M tokens (output includes thinking; cached input is 95% off).
# Introductory rate through 2026-11-30, standard rate from 2026-12-01. Check ai.google.dev/gemini-api/docs/pricing.
PRICES = [("2026-11-30", {"in": 2.0, "out": 10.0}),       # (last day the rate applies, rates)
          ("9999-12-31", {"in": 4.0, "out": 20.0})]
CACHED_DISCOUNT = 0.95

BASE  = os.environ.get("SOCCERLENS_BASE", "/content/drive/MyDrive")
LOCAL = "/content" if os.path.isdir("/content") else os.getcwd()
REPO        = f"{LOCAL}/SoccerExplainability"
DATA_ROOT   = f"{BASE}/SoccerNet_720p"
OUT_DIR     = f"{BASE}/SoccerLens_LLM_annotations_gemini4_induced"
FRAME_CACHE = f"{LOCAL}/soccerlens_frames"
FRAMES_TAR  = f"{BASE}/soccerlens_frames.tar"
GRID = 1000

LABELS = ["small label", "large label", "visual cue"]
CAT_ID = {"small label": 1, "large label": 2, "visual cue": 3}
TIER_NAME  = {"small label": "PRIMARY", "large label": "SECONDARY", "visual cue": "COMMON"}
TIER_COLOR = {"small label": ("red", (230, 30, 30)), "large label": ("blue", (40, 90, 255)), "visual cue": ("yellow", (250, 220, 0))}
TIER_DEF = {
    "small label": "PRIMARY cue: the most salient visual features relevant to the event, in a highly localized manner.",
    "large label": "SECONDARY cue: less localized than primary cues; if a frame has multiple primary cues, "
                   "they should be contained within a secondary cue.",
    "visual cue":  "COMMON cue: visual features temporally associated with the event that do not represent "
                   "the event itself (distractors that co-occur with it).",
}
CUE_SCHEMA = {
    "corner": {"small label": "corner flag, player heading to the corner flag, corner play",
               "large label": "larger box around the corner arc area",
               "visual cue":  "players grouped close to the net, net or goalkeeper, TV panel"},
    "goal": {"small label": "shot on goal, ball, assist",
             "large label": "goalmouth area, goalkeeper, net",
             "visual cue":  "player celebration, scoreboard update, referee pointing to center circle, fans cheering"},
    "yellow card": {"small label": "referee raising the yellow card, player receiving the yellow card",
                    "large label": "player(s) involved in the foul and referee grouped together",
                    "visual cue":  "foul leading to card, TV panel, player(s) hurt"},
    "foul with no card": {"small label": "foul action, referee",
                          "large label": "player(s) involved in foul and the referee",
                          "visual cue":  "injured player, free kick, another irrelevant foul within the 30-second window"},
    "lead to corner": {"small label": "final touch before ball crosses the line, and the ball",
                       "large label": "player and ball as ball leaves the field",
                       "visual cue":  "corner play, referee"},
    "free kick": {"small label": "kicker, ball, referee, defensive wall of players",
                  "large label": "kicker and defensive wall and ball together",
                  "visual cue":  "foul leading to free kick"},
    "substitution": {"small label": "4th official's board, players entering or leaving the field",
                     "large label": "both players involved in the substitution",
                     "visual cue":  "TV panel"},
    "injury": {"small label": "medical staff, referee, player injured",
               "large label": "injured player and medical staff",
               "visual cue":  "TV panel, players around the injured player"},
    "foul lead to penalty": {"small label": "foul action, player(s) on ground, referee",
                             "large label": "players involved in foul and the referee",
                             "visual cue":  "penalty play"},
    "penalty": {"small label": "shot on goal and goalmouth area",
                "large label": "goalmouth area, net and the kicker altogether",
                "visual cue":  "referee, foul leading to penalty"},
    "throw in": {"small label": "player throwing the ball and the player receiving the ball",
                 "large label": "both players throwing and receiving the ball",
                 "visual cue":  "out-of-play action, referee"},
    "second yellow card": {"small label": "referee raising the cards, referee pointing to tunnel",
                           "large label": "player(s) involved in the foul and referee grouped together",
                           "visual cue":  "foul leading to card, TV panel, player receiving card"},
    "red card": {"small label": "referee raising the red card, referee pointing to tunnel",
                 "large label": "player(s) involved in the foul and referee grouped together",
                 "visual cue":  "foul leading to red card, TV panel, player receiving card"},
}

assert MODE in ("induce", "annotate", "both", "frames") and INDUCE_FOLDS in (1, 2)
import numpy as np, cv2
from PIL import Image, ImageDraw

def sh(*cmd):
    subprocess.run(list(cmd), check=True)

assert os.path.isdir(BASE), f"{BASE} not found: mount Drive first (drive.mount('/content/drive'))"
if not os.path.exists(REPO):
    sh("git", "clone", "-q", "<ANONYMIZED_REPO_URL>", REPO)
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FRAME_CACHE, exist_ok=True)
if os.path.exists(FRAMES_TAR) and not os.listdir(FRAME_CACHE):
    print(f"unpacking {FRAMES_TAR} ...")
    with tarfile.open(FRAMES_TAR) as tf:
        tf.extractall(FRAME_CACHE, filter="data")
print(f"outputs -> {OUT_DIR} | frames -> {FRAME_CACHE} ({len(os.listdir(FRAME_CACHE))} clips cached)")

# ----------------------------- data (same as the Qwen script) -----------------------------
sel = json.load(open(f"{REPO}/train_data/json/selected_videos_for_annotations.json"))
coco = json.load(open(f"{REPO}/annotations-coco.json"))
imgs_by_video = defaultdict(dict)
for im in coco["images"]:
    imgs_by_video[im["video_id"]][im["second"]] = im
gt_by_image = defaultdict(list)
for a in coco["annotations"]:
    gt_by_image[a["image_id"]].append(a)
img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

clips, _seen = [], set()
for item in sel:
    v, e = item["video"], item["caption"]
    key = f"{v}#{e}" if f"{v}#{e}" in imgs_by_video else v
    if key in _seen:
        continue
    _seen.add(key); clips.append({"key": key, "video": v, "event": e})
assert not ({c["event"] for c in clips} - set(CUE_SCHEMA))
if ONLY_EVENTS:
    clips = [c for c in clips if c["event"] in set(ONLY_EVENTS)]
print(f"{len(clips)} clips | {len(imgs_by_video)} annotated clips in COCO")
clip_by_key = {c["key"]: c for c in clips}
by_event = defaultdict(list)
for c in clips:
    by_event[c["event"]].append(c["key"])

def gt_boxes(image_id):
    w, h = img_wh[image_id]
    out = []
    for a in gt_by_image[image_id]:
        if a.get("no_roi") or a.get("label") not in CAT_ID:
            continue
        x, y, bw, bh = a["bbox"]
        out.append((a["label"], x / w, y / h, bw / w, bh / h))
    return out

_rng = random.Random(SEED); fold_of = {}
for e, keys in sorted(by_event.items()):
    ks = sorted(keys); _rng.shuffle(ks)
    for i, k in enumerate(ks):
        fold_of[k] = i % INDUCE_FOLDS

def source_clips(event, fold):
    return sorted(by_event[event]) if INDUCE_FOLDS == 1 else sorted(k for k in by_event[event] if fold_of[k] != fold)

# ----------------------------- frames (same sampling and cache as the Qwen script) -----------------------------
def middle_indices(vlen, num_frames=30):
    acc = min(num_frames, vlen)
    iv = np.linspace(0, vlen, acc + 1).astype(int)
    idx = [(iv[i] + iv[i + 1] - 1) // 2 for i in range(acc)]
    return idx + [idx[-1]] * (num_frames - len(idx))

def video_source(video_id):
    league_season, match, fname = video_id.split("/")
    league, season = league_season.rsplit("_", 1)
    half, mm, ss = map(int, re.search(r"(\d)_(\d+)_(\d+)\.mp4$", fname).groups())
    return f"{league}/{season}/{match}/{half}_720p.mkv", half, mm, ss

def cache_dir(video_id):
    return f"{FRAME_CACHE}/{hashlib.md5(video_id.encode()).hexdigest()}"

def frames_cached(video_id):
    return all(os.path.exists(f"{cache_dir(video_id)}/{k:02d}.jpg") for k in range(30))

def cut_frames(video_id, src, threads=0):
    """Same ffmpeg settings as the Qwen run (libx264 crf 18), so the frames are identical."""
    rel, half, mm, ss = video_source(video_id)
    clip = f"{LOCAL}/_clip_{hashlib.md5(video_id.encode()).hexdigest()}.mp4"   # unique: clips are cut in parallel
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(mm * 60 + ss - 15, 0)), "-i", src,
       "-t", "30", "-an", "-c:v", "libx264", "-crf", "18", "-threads", str(threads), clip)
    cap = cv2.VideoCapture(clip); vlen = 0
    while cap.grab():
        vlen += 1
    cap.release()
    picks = middle_indices(vlen); need = set(picks); got = {}
    cap = cv2.VideoCapture(clip)
    for fi in range(vlen):
        ok, f = cap.read()
        if not ok:
            break
        if fi in need:
            got[fi] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    os.remove(clip)
    d = cache_dir(video_id); os.makedirs(d, exist_ok=True)
    for k, p in enumerate(picks):
        got[p].save(f"{d}/{k:02d}.jpg", format="JPEG", quality=95)

def pack_frames():
    tmp = f"{LOCAL}/_frames.tar"
    with tarfile.open(tmp, "w") as tf:
        for d in sorted(os.listdir(FRAME_CACHE)):
            tf.add(f"{FRAME_CACHE}/{d}", arcname=d)
    shutil.copy(tmp, FRAMES_TAR); os.remove(tmp)            # one sequential write to Drive

def ensure_frames():
    """Cut missing frames one source video at a time: copy the match half from Drive to local disk with one
    sequential read (ffmpeg seeking inside a file on the Drive mount is very slow), then cut all its clips in
    parallel. The cache is packed to Drive every few videos, so a runtime reset loses little."""
    need = defaultdict(list)
    for v in sorted({c["video"] for c in clips}):
        if not frames_cached(v):
            need[video_source(v)[0]].append(v)
    if not need:
        return
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg missing: !apt-get -qq install -y ffmpeg")
    n_cpu = os.cpu_count() or 2
    n_par = max(min(n_cpu // 2, 8), 1)                     # clips cut at once, each ffmpeg gets n_cpu / n_par threads
    print(f"\n=== cutting frames for {sum(map(len, need.values()))} clips from {len(need)} videos "
          f"({n_cpu} CPUs, {n_par} clips at a time) ===")
    t0 = time.time(); local_src = f"{LOCAL}/_src.mkv"
    for i, (rel, vids) in enumerate(sorted(need.items()), 1):
        src = f"{DATA_ROOT}/{rel}"
        if not os.path.exists(src):
            raise SystemExit(f"video missing: {src}\nPut soccerlens_frames.tar in {BASE}, "
                             "or a shortcut to the shared SoccerNet_720p folder in My Drive.")
        t1 = time.time(); shutil.copy(src, local_src); t_copy = time.time() - t1
        with ThreadPoolExecutor(max_workers=n_par) as ex:
            list(ex.map(lambda v: cut_frames(v, local_src, max(n_cpu // n_par, 1)), vids))
        os.remove(local_src)
        print(f"  video {i}/{len(need)} | {len(vids)} clip(s) | copy {t_copy:.0f}s | "
              f"total {(time.time() - t0) / 60:.1f} min", flush=True)
        if i % 5 == 0 and i < len(need):
            pack_frames()
    pack_frames()
    print(f"packed the frames into {FRAMES_TAR}")

ensure_frames()
if MODE == "frames":
    sys.exit(0)

_frame_lock = threading.Lock(); _frame_mem = {}
def frames_of(video_id):
    """30 frames of a clip as (jpeg bytes, (W, H))."""
    with _frame_lock:
        if video_id in _frame_mem:
            return _frame_mem[video_id]
    out = []
    for k in range(30):
        b = open(f"{cache_dir(video_id)}/{k:02d}.jpg", "rb").read()
        out.append((b, Image.open(io.BytesIO(b)).size))
    with _frame_lock:
        if len(_frame_mem) >= 64:
            _frame_mem.pop(next(iter(_frame_mem)))
        _frame_mem[video_id] = out
    return out

def seconds_of(key):
    return sorted(imgs_by_video[key]) if SECONDS_MODE == "annotated" else list(range(30))

# ----------------------------- Gemini client -----------------------------
from google import genai
from google.genai import types
assert os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), \
    "set GEMINI_API_KEY first: os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
# Two ways in, picked from the environment:
#  - Vertex AI / Gemini Enterprise: GOOGLE_GENAI_USE_ENTERPRISE (or _VERTEXAI) = true, or a Google Cloud key ("AQ...").
#    The key is passed explicitly: if it only sits in the environment next to GOOGLE_CLOUD_PROJECT, the SDK prefers
#    the project and calls Vertex with Colab's own login instead of the key, which is refused (403).
#  - Gemini Developer API (AI Studio key, "AIza..."): everything else.
_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
# No request may hang forever: without a timeout one stuck connection blocks the run with no output.
_HTTP = types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000)
_vertex = (any(os.environ.get(v, "").lower() in ("true", "1") for v in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"))
           or (_key or "").startswith("AQ."))
if _vertex:
    # Key + project + location all explicit: requests go to projects/<project>/locations/global/... with key auth.
    # With the key alone the SDK uses express mode, which defaults to us-central1, where Argon is not served (404).
    _proj, _loc = os.environ.get("GOOGLE_CLOUD_PROJECT"), os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    if not _proj:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is not set: run the cell that sets GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT, "
                         "GOOGLE_CLOUD_LOCATION and GOOGLE_GENAI_USE_ENTERPRISE, then run this script again.")
    client = genai.Client(vertexai=True, api_key=_key, project=_proj, location=_loc, http_options=_HTTP)
    print(f"using Vertex AI (Gemini Enterprise) with the API key | project {_proj} | location {_loc}")
    if API == "auto":
        API = "generate_content"          # Vertex serves generateContent; the Interactions API is a Developer API surface
else:
    client = genai.Client(vertexai=False, api_key=_key, http_options=_HTTP)
    print("using the Gemini Developer API (AI Studio key)")
_usage = defaultdict(int); _usage_lock = threading.Lock()

# A request is a list of parts: ("text", str) or ("image", jpeg_bytes, resolution)
def _call_interactions(parts, think, max_out):
    inp = []
    for p in parts:
        if p[0] == "text":
            inp.append({"type": "text", "text": p[1]})
        else:
            img = {"type": "image", "data": base64.b64encode(p[1]).decode(), "mime_type": "image/jpeg"}
            if p[2]:
                img["resolution"] = p[2]
            inp.append(img)
    r = client.interactions.create(model=MODEL_ID, input=inp, store=False,
                                   generation_config={"thinking_level": think, "max_output_tokens": max_out, "seed": SEED})
    u = getattr(r, "usage", None)
    return r.output_text or "", {"in": getattr(u, "total_input_tokens", 0) or 0,
                                 "cached": getattr(u, "total_cached_tokens", 0) or 0,
                                 "thought": getattr(u, "total_thought_tokens", 0) or 0,
                                 "out": (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)}

def _call_generate(parts, think, max_out):
    contents = [types.Part.from_text(text=p[1]) if p[0] == "text" else types.Part.from_bytes(data=p[1], mime_type="image/jpeg")
                for p in parts]
    cfg = types.GenerateContentConfig(max_output_tokens=max_out, seed=SEED,
                                      thinking_config=types.ThinkingConfig(thinking_level=think))
    r = client.models.generate_content(model=MODEL_ID, contents=contents, config=cfg)
    u = r.usage_metadata
    return r.text or "", {"in": getattr(u, "prompt_token_count", 0) or 0,
                          "cached": getattr(u, "cached_content_token_count", 0) or 0,
                          "thought": getattr(u, "thoughts_token_count", 0) or 0,
                          "out": (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)}

_CALLERS = {"interactions": _call_interactions, "generate_content": _call_generate}

def gemini(parts, think, max_out):
    """One request with retry and exponential backoff on rate limits and server errors."""
    for attempt in range(MAX_RETRIES):
        try:
            text, u = _CALLERS[API](parts, think, max_out)
            with _usage_lock:
                for k in ("in", "cached", "thought", "out"):
                    _usage[k] += u[k]
                _usage["requests"] += 1
                _usage["cost"] += request_cost(u)
            return text
        except Exception as ex:
            msg = str(ex)
            if attempt == MAX_RETRIES - 1 or not re.search(r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|timed? ?out|timeout|Timeout|ReadError|ConnectError|RemoteProtocol", msg, re.I):
                raise
            time.sleep(min(2 ** attempt * 5, 120) + random.random() * 3)

def rates_on(day=None):
    day = day or time.strftime("%Y-%m-%d", time.gmtime())
    return next(r for last, r in PRICES if day <= last)

def request_cost(u, rates=None):
    """USD for one request's usage at the rate in force when it was made (or at `rates`)."""
    r = rates or rates_on()
    fresh = max(u["in"] - u["cached"], 0)
    return (fresh + u["cached"] * (1 - CACHED_DISCOUNT)) / 1e6 * r["in"] + u["out"] / 1e6 * r["out"]

def usage_line():
    return (f"{_usage['requests']} requests | {_usage['in'] / 1e6:.2f}M input tokens | "
            f"{_usage['out'] / 1e6:.2f}M output+thinking tokens | ${_usage['cost']:.2f} so far")

def print_cost_report():
    u = dict(_usage)
    if not u.get("requests"):
        print("\nno API requests were made in this run, so there is nothing to bill"); return
    fresh = max(u["in"] - u["cached"], 0)
    print("\n===================== cost of this run =====================")
    print(f"model {MODEL_ID} | {u['requests']} requests")
    print(f"input        {u['in']:>14,} tokens  ({u['cached']:,} of them cached)")
    print(f"output       {u['out']:>14,} tokens  ({u['thought']:,} of them thinking)")
    print(f"billed at the rates in force when each request was made:  ${u['cost']:.2f}")
    for last, r in PRICES:
        c = (fresh + u["cached"] * (1 - CACHED_DISCOUNT)) / 1e6 * r["in"] + u["out"] / 1e6 * r["out"]
        when = f"through {last}" if last != "9999-12-31" else f"after {PRICES[-2][0]}"
        print(f"  same tokens at ${r['in']:g} in / ${r['out']:g} out per 1M ({when}):  ${c:.2f}")
    print("Estimate from list prices; the billing page in Google Cloud is the final figure.")
    print("=============================================================")

import atexit
atexit.register(print_cost_report)          # printed at the end of every run, also after an error or MODE = 'induce'

# Smoke test: pick the API surface that this model answers on.
_probe = [("text", 'Reply with the JSON array [] and nothing else.')]
for cand in (["interactions", "generate_content"] if API == "auto" else [API]):
    try:
        API = cand; t = gemini(_probe, "low", 256)
        print(f"model {MODEL_ID} answers through the {cand} API: {t.strip()[:40]!r}")
        break
    except Exception as ex:
        print(f"  {cand} API failed: {str(ex)[:300]}")
else:
    raise SystemExit(f"{MODEL_ID} did not answer on any API surface. Check the model ID in AI Studio and your key's access.")

# File names record whether the written criteria were used, so runs with and without them never mix.
_crit = ("_crit" if USE_CRITERIA_IN_INDUCTION else "_nocrit")
tag = MODEL_ID + _crit + ("-annotcrit" if USE_CRITERIA_IN_ANNOTATION else "") + f"_think-{ANNOT_THINK}"
PROMPTS_JSON = f"{OUT_DIR}/{MODEL_ID}{_crit}_induced_prompts_folds{INDUCE_FOLDS}.json"
PRED_JSONL   = f"{OUT_DIR}/{tag}_induced_folds{INDUCE_FOLDS}_raw.jsonl"

# ----------------------------- boxes in Gemini's format -----------------------------
def to_box2d(x, y, w, h):
    """normalized (x, y, w, h) -> Gemini box_2d [ymin, xmin, ymax, xmax] on the 0-1000 grid"""
    return [round(GRID * y), round(GRID * x), round(GRID * (y + h)), round(GRID * (x + w))]

def boxes_json(bs):
    return json.dumps([{"label": l, "box_2d": to_box2d(x, y, w, h)}
                       for l, x, y, w, h in sorted(bs, key=lambda b: (LABELS.index(b[0]), b[1], b[2]))])

def parse(text):
    """Model answer -> list of {"label", "bbox_2d": [x1, y1, x2, y2]} on the 0-1000 grid (Qwen-run convention), or None."""
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
        if not isinstance(o, dict):
            continue
        lab, bb = o.get("label"), o.get("box_2d")
        if lab in CAT_ID and isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                ymin, xmin, ymax, xmax = [float(v) for v in bb]
            except (TypeError, ValueError):
                continue
            out.append({"label": lab, "bbox_2d": [xmin, ymin, xmax, ymax]})
    return out

# ----------------------------- stage 1: prompt induction -----------------------------
def example_jpeg(jpeg, bs):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    W, H = img.size
    if W > INDUCE_MAX_SIDE:
        img = img.resize((INDUCE_MAX_SIDE, max(round(H * INDUCE_MAX_SIDE / W), 1)))
    w, h = img.size; d = ImageDraw.Draw(img)
    for l, x, y, bw, bh in sorted(bs, key=lambda b: -b[3] * b[4]):
        x0, y0 = min(max(x * w, 0), w - 1), min(max(y * h, 0), h - 1)
        x1, y1 = min(max((x + bw) * w - 1, x0), w - 1), min(max((y + bh) * h - 1, y0), h - 1)
        d.rectangle([x0, y0, x1, y1], outline=TIER_COLOR[l][1], width=3)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def pick_seconds(key, n):
    secs = sorted(imgs_by_video[key])
    if len(secs) <= n:
        return secs
    return sorted({secs[i] for i in np.linspace(0, len(secs) - 1, n).round().astype(int)})

def induction_request(event, keys):
    sch = CUE_SCHEMA[event]
    per_clip = max(INDUCE_MAX_IMAGES // max(len(keys), 1), 1)
    plan = [(k, s) for k in keys for s in pick_seconds(k, per_clip)][:INDUCE_MAX_IMAGES]
    if USE_CRITERIA_IN_INDUCTION:
        cats = ("Human annotators drew bounding boxes for three categories of visual cue:\n"
                + "\n".join(f'- "{l}" = {TIER_DEF[l]} Written criteria for "{event}": {sch[l]}.' for l in LABELS) + "\n\n")
        legend = ", ".join(f'{TIER_COLOR[l][0]} = {TIER_NAME[l]} ("{l}")' for l in LABELS)
    else:   # only the category names: what each category means must be deduced from the boxes
        cats = ('Human annotators drew bounding boxes for three categories of visual cue, named "small label", '
                '"large label" and "visual cue". You are not told what each category means; work it out from the examples.\n\n')
        legend = ", ".join(f'{TIER_COLOR[l][0]} = "{l}"' for l in LABELS)
    parts = [("text",
        f'You are helping to write annotation instructions for soccer broadcast frames of the event "{event}".\n'
        + cats +
        f"Below are {len(plan)} annotated frames taken from {len(keys)} different 30-second clips of this event "
        "(the event happens around second 15 of each clip). On every image the human boxes are drawn: " + legend
        + f". The same boxes are listed after the image as box_2d = [ymin, xmin, ymax, xmax] on a 0-{GRID} grid. "
        "A frame followed by [] is one where the annotators decided that no cue is visible.\n")]
    last = None
    for i, (k, s) in enumerate(plan, 1):
        bs = gt_boxes(imgs_by_video[k][s]["id"])
        clip_no = keys.index(k) + 1
        head = f"Frame {i}: clip {clip_no}, second {s}." if k == last else f"--- clip {clip_no} ---\nFrame {i}: clip {clip_no}, second {s}."
        parts += [("text", head), ("image", example_jpeg(frames_of(clip_by_key[k]["video"])[s][0], bs), INDUCE_RES),
                  ("text", "Human boxes: " + boxes_json(bs))]
        last = k
    parts.append(("text",
        ("Study how the annotators applied the written criteria: " if USE_CRITERIA_IN_INDUCTION else
         "Study the annotations to work out what each category stands for in this event: ")
        + "which people and objects they boxed for each category "
        "and which they left out, how tightly and how far each box extends (whole body, a group, an area of the pitch, a "
        "TV graphic), how many boxes of each category a frame usually gets, how the categories nest, and in which kinds "
        "of frame they boxed nothing. Pay particular attention to the frames with no boxes and to what the annotators "
        "did NOT box: an annotator who boxes too much is as wrong as one who misses cues.\n"
        f'Then write the instructions that would let an annotator who sees ONE frame of a "{event}" clip, and nothing '
        "else, reproduce these boxes as closely as possible. Requirements:\n"
        '- Address the annotator directly and be concrete about what to box for each of the three categories, using the '
        'exact category names "small label", "large label" and "visual cue".\n'
        "- State the box extent, the usual number of boxes per category, and the nesting rule if there is one.\n"
        "- State clearly when to return no boxes at all, and the mistakes to avoid.\n"
        f"- Give typical box sizes on the 0-{GRID} grid only where they are consistent across the frames.\n"
        "- Do not mention these example frames, clips, colors or their coordinates: the annotator will not see them.\n"
        f"- At most {INDUCE_MAX_WORDS} words.\n"
        "Put the final instructions between <prompt> and </prompt>. Write nothing after </prompt>."))
    return parts, len(plan), per_clip

def extract_prompt(text):
    m = re.search(r"<prompt>(.*?)</prompt>", text or "", flags=re.S)
    out = m.group(1).strip() if m else ""
    return out if len(out.split()) >= 40 else ""

induced = json.load(open(PROMPTS_JSON)) if os.path.exists(PROMPTS_JSON) and not REINDUCE else {}
for v in induced.values():
    if not (v.get("edited") and v.get("prompt")):          # set "edited": true on a prompt you changed by hand
        v["prompt"] = extract_prompt(v.get("raw")); v["ok"] = bool(v["prompt"])

if MODE in ("induce", "both"):
    todo = [(e, f) for e in sorted(by_event) for f in range(INDUCE_FOLDS) if not induced.get(f"{e}|{f}", {}).get("ok")]
    print(f"\n=== stage 1: {len(todo)} prompts to deduce ({len(by_event)} events x {INDUCE_FOLDS} fold(s)) ===")

    def induce_one(e, f):
        keys = source_clips(e, f)
        if not keys:
            return e, f, None
        parts, n_img, per_clip = induction_request(e, keys)
        t0 = time.time(); raw, prompt = "", ""
        for attempt in range(2):                           # an answer without usable <prompt> tags is asked again
            raw = gemini(parts, INDUCE_THINK, INDUCE_MAX_OUT)
            prompt = extract_prompt(raw)
            if prompt:
                break
        return e, f, {"prompt": prompt, "ok": bool(prompt), "clips": len(keys), "frames": n_img,
                      "frames_per_clip": per_clip, "raw": raw, "seconds": round(time.time() - t0)}

    with ThreadPoolExecutor(max_workers=min(4, max(len(todo), 1))) as ex:
        for fut in as_completed([ex.submit(induce_one, e, f) for e, f in todo]):
            e, f, rec = fut.result()
            if rec is None:
                print(f"  {e} fold {f}: no source clips, the base prompt will be used"); continue
            induced[f"{e}|{f}"] = rec
            json.dump(induced, open(PROMPTS_JSON, "w"), indent=1)
            st = f"{len(rec['prompt'].split())} words" if rec["ok"] else "FAILED, the base prompt will be used"
            print(f"  {e} fold {f}: {rec['clips']} clips, {rec['frames']} frames -> {st} | {rec['seconds']}s")
    for e in sorted(by_event):
        print(f'\n----- deduced instructions for "{e}" -----\n' + (induced.get(f"{e}|0", {}).get("prompt") or "(none)"))
    print(f"\n{sum(v['ok'] for v in induced.values())} of {len(induced)} prompts usable | {usage_line()}")

if MODE == "induce":
    sys.exit(0)
if not induced:
    raise SystemExit(f"no deduced prompts at {PROMPTS_JSON}: run MODE induce or both first")

# ----------------------------- stage 2: annotation -----------------------------
def build_prompt(clip):
    event = clip["event"]; sch = CUE_SCHEMA[event]
    if USE_CRITERIA_IN_ANNOTATION:
        defs = "\n".join(f'- "{k}" = {TIER_DEF[k]} For "{event}", box: {sch[k]}.' for k in LABELS)
        cats = f"Annotate the visual cues for this event with bounding boxes in three categories:\n{defs}\n\n"
    else:
        cats = ('Annotate the visual cues for this event with bounding boxes in three categories: '
                '"small label", "large label" and "visual cue".\n\n')
    p = f'This frame is from a 30-second soccer broadcast clip of the event: "{event}".\n' + cats
    guide = induced.get(f"{event}|{fold_of[clip['key']]}", {}).get("prompt")
    p += (f'Annotation instructions for "{event}" (follow them exactly):\n{guide}\n\n' if guide else
          "Box only cues that are visible in THIS frame. Use as many boxes per category as needed.\n")
    p += ('Return ONLY a JSON array, one object per box: [{"label": "<small label|large label|visual cue>", '
          f'"box_2d": [ymin, xmin, ymax, xmax]}}], coordinates normalized to 0-{GRID} relative to the image. '
          "If no cue for this event is visible in the frame, return [].")
    return p

def prompt_md5(clip):
    return hashlib.md5(build_prompt(clip).encode()).hexdigest()

def read_jsonl(path):
    """Rows of a JSONL file; a line cut off by a disconnect is skipped (that frame is simply redone)."""
    rows = []
    if os.path.exists(path):
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return rows

run_clips = clips[:MAX_CLIPS] if MAX_CLIPS else clips
_old = read_jsonl(PRED_JSONL); _cur = {c["key"]: prompt_md5(c) for c in clips}
_keep = [r for r in _old if r.get("prompt_md5") == _cur.get(r["video_id"]) and r.get("boxes") is not None]
_nlines = sum(1 for _ in open(PRED_JSONL)) if os.path.exists(PRED_JSONL) else 0
if len(_keep) < _nlines:      # also rewrites the file when its last line was cut off, so new rows start clean
    shutil.copy(PRED_JSONL, f"{PRED_JSONL}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(PRED_JSONL, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in _keep)
    print(f"{_nlines - len(_keep)} saved frames were cut off, unparseable or made with another prompt: redoing them")
done = {(r["video_id"], r["second"]) for r in _keep}
jobs = [(c, s) for c in run_clips for s in seconds_of(c["key"]) if (c["key"], s) not in done]
print(f"\n=== stage 2: {len(run_clips)} clips | {len(done)} frames done | {len(jobs)} to do | {N_WORKERS} workers ===")
print("prompt for the first clip:\n" + "-" * 60 + "\n" + build_prompt(run_clips[0]) + "\n" + "-" * 60)

_write_lock = threading.Lock()
def annotate_one(clip, s):
    jpeg, (W, H) = frames_of(clip["video"])[s]
    try:
        txt = gemini([("image", jpeg, ANNOT_RES), ("text", build_prompt(clip))], ANNOT_THINK, ANNOT_MAX_OUT)
    except Exception as ex:
        txt = f"__ERROR__ {ex}"
    return {"video_id": clip["key"], "second": s, "event": clip["event"], "fold": fold_of[clip["key"]],
            "prompt_md5": prompt_md5(clip), "width": W, "height": H, "raw": txt, "boxes": parse(txt)}

if jobs:
    t0 = time.time(); n = n_fail = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex, open(PRED_JSONL, "a") as fh:
        for fut in as_completed([ex.submit(annotate_one, c, s) for c, s in jobs]):
            row = fut.result(); n += 1; n_fail += row["boxes"] is None
            with _write_lock:
                fh.write(json.dumps(row) + "\n"); fh.flush()
            if n % 100 == 0 or n == len(jobs) or len(jobs) - n < 100 and n % 10 == 0:
                print(f"  {n}/{len(jobs)} frames | {n_fail} unparseable | {(time.time() - t0) / 60:.1f} min | {usage_line()}")

# ----------------------------- scoring (identical to the Qwen run, grid coordinates) -----------------------------
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0

def to_norm(row):
    out = []
    for b in row["boxes"] or []:
        x1, y1, x2, y2 = b["bbox_2d"]
        x1, x2 = sorted((min(max(x1 / GRID, 0.0), 1.0), min(max(x2 / GRID, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1 / GRID, 0.0), 1.0), min(max(y2 / GRID, 0.0), 1.0)))
        if x2 - x1 > 0 and y2 - y1 > 0:
            out.append((b["label"], [x1, y1, x2 - x1, y2 - y1]))
    return out

def agreement(rows, keys=None):
    stats = {l: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for l in LABELS}
    noroi = {"both_empty": 0, "human_empty_llm_boxes": 0, "human_boxes_llm_empty": 0, "both_boxes": 0}
    n = 0
    for r in rows:
        im = imgs_by_video[r["video_id"]].get(r["second"])
        if r["boxes"] is None or im is None or (keys is not None and r["video_id"] not in keys):
            continue
        n += 1; gt = gt_boxes(im["id"]); pr = to_norm(r)
        noroi[("both_empty" if not pr else "human_empty_llm_boxes") if not gt else
              ("human_boxes_llm_empty" if not pr else "both_boxes")] += 1
        for l in LABELS:
            g = [list(b[1:]) for b in gt if b[0] == l]; p = [b for lab, b in pr if lab == l]
            pairs = sorted(((iou(pb, gb), i, j) for i, pb in enumerate(p) for j, gb in enumerate(g)), reverse=True)
            up, ug = set(), set()
            for v, i, j in pairs:
                if v < 0.5:
                    break
                if i in up or j in ug:
                    continue
                up.add(i); ug.add(j); stats[l]["ious"].append(v)
            stats[l]["tp"] += len(up); stats[l]["fp"] += len(p) - len(up); stats[l]["fn"] += len(g) - len(ug)
    res = {}
    for l, s in stats.items():
        P = s["tp"] / max(s["tp"] + s["fp"], 1); R = s["tp"] / max(s["tp"] + s["fn"], 1)
        res[l] = {"P": P, "R": R, "F1": 2 * P * R / max(P + R, 1e-9),
                  "mIoU": float(np.mean(s["ious"])) if s["ious"] else 0.0, "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}
    used = [l for l in LABELS if res[l]["tp"] + res[l]["fp"] + res[l]["fn"] > 0]
    return {"frames": n, "macro_f1": float(np.mean([res[l]["F1"] for l in used])) if used else 0.0,
            "tiers": res, "used_tiers": used, "no_roi": noroi}

rows = read_jsonl(PRED_JSONL)
preds = []
for r in rows:
    if r["boxes"] is None:
        continue
    im = imgs_by_video[r["video_id"]].get(r["second"])
    for lab, b in to_norm(r):
        preds.append({"image_id": im["id"] if im else f'{r["video_id"]}_{r["second"]}', "category_id": CAT_ID[lab],
                      "label": lab, "bbox": b, "score": 1.0})
json.dump(preds, open(f"{OUT_DIR}/{tag}_induced_folds{INDUCE_FOLDS}_coco_preds.json", "w"))

a = agreement(rows)
regime = ("prompts deduced from ALL clips of each event, including the ones scored here: an upper bound, not held-out"
          if INDUCE_FOLDS == 1 else "cross-fitted: each clip annotated with a prompt deduced from other clips only")
print(f"\nAgreement with human boxes (IoU>=0.5) | {regime}\n{a['frames']} frames | macro-F1 {a['macro_f1']:.3f} | "
      f"{sum(r['boxes'] is None for r in rows)} unparseable")
print(f"{'label':<12}{'P':>7}{'R':>7}{'F1':>7}{'mIoU':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'pred/human':>12}")
for l in LABELS:
    t = a["tiers"][l]
    print(f"{l:<12}{t['P']:7.3f}{t['R']:7.3f}{t['F1']:7.3f}{t['mIoU']:7.3f}{t['tp']:6d}{t['fp']:6d}{t['fn']:6d}"
          f"{(t['tp'] + t['fp']) / max(t['tp'] + t['fn'], 1):11.2f}x")
nr = a["no_roi"]; he = nr["both_empty"] + nr["human_empty_llm_boxes"]
print(f"frame-level: {nr} | model abstains on {nr['both_empty']}/{he} frames the humans left empty")

per_event = {e: agreement(rows, keys=set(by_event[e])) for e in sorted(by_event)}
fmt = lambda e, t: "   n/a" if t not in per_event[e]["used_tiers"] else f"{per_event[e]['tiers'][t]['F1']:6.3f}"
print(f"\n{'event':<22}{'frames':>7}{'macroF1':>9}{'primary':>9}{'second.':>9}{'common':>8}")
for e, ev in per_event.items():
    print(f"{e:<22}{ev['frames']:>7}{ev['macro_f1']:>9.3f}{fmt(e, 'small label'):>9}{fmt(e, 'large label'):>9}"
          f"{fmt(e, 'visual cue'):>8}")
print(f"{'ALL':<22}{a['frames']:>7}{a['macro_f1']:>9.3f}")
print(usage_line())
json.dump({"agreement": a, "per_event": per_event, "regime": regime, "induce_folds": INDUCE_FOLDS, "model": MODEL_ID,
           "api": API, "criteria_in_induction": USE_CRITERIA_IN_INDUCTION,
           "criteria_in_annotation": USE_CRITERIA_IN_ANNOTATION, "induce_think": INDUCE_THINK, "annot_think": ANNOT_THINK, "induce_max_images": INDUCE_MAX_IMAGES,
           "seed": SEED, "usage": dict(_usage), "prices": PRICES, "prompts_file": PROMPTS_JSON},
          open(f"{OUT_DIR}/{tag}_induced_folds{INDUCE_FOLDS}_agreement.json", "w"), indent=1)
print(f"saved to {OUT_DIR}")
# printed here as well: when the script runs inside the notebook kernel, atexit only fires when the kernel stops
atexit.unregister(print_cost_report)
print_cost_report()
