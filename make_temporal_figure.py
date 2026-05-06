"""
Build a figure showing Chefer attributions at specific timestamps.

Layout:
    Columns : Original | MatchVision | SigLIP | SoccerMaster
    Rows    : one per timestamp

Run:
    python make_temporal_figure.py"""

import io
import json
import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIZ_ROOT = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLensVisualizations-2-5-2026-chefer-full/SoccerLensVisualizations-2-5-2026"
)
MATCH_DIR = "2015-10-24 - 17-00 West Ham 2 - 1 Chelsea"
CHEFER_VIDEO_NAME = "1_19_05.mp4"

ORIGINAL_VIDEO = (
    r"/mnt/c/Users/z004kjmt/Downloads/SoccerLens-for-annotation"
    r"/SoccerLens-for-annotation/england_epl_2015-2016"
    r"/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea/1_19_05.mp4"
)
ORIGINAL_FPS = 30  # original clip fps

CHEFER_VIDEOS = {
    "MatchVision":  os.path.join(VIZ_ROOT, "MatchVision",   "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
    "SigLIP":       os.path.join(VIZ_ROOT, "SigLip",        "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
    "SoccerMaster": os.path.join(VIZ_ROOT, "SoccerMaster",  "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
}

TIMESTAMPS = [25]   # seconds into the clip

OUTPUT_PATH = os.path.join(VIZ_ROOT, "throw_in_chefer_figure.png")

ORIGINAL_BASE = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLens-for-annotation/SoccerLens-for-annotation"
)

# Multi-clip rows: each entry defines one row
# (label, video_rel_path, timestamp_sec)
ROW_ENTRIES = [
    (
        "Throw In",
        "england_epl_2015-2016/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea/1_19_05.mp4",
        25,
    ),
    (
        "Foul with no Card",
        "germany_bundesliga_2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund/2_32_19.mp4",
        13,
    ),
]

MULTI_OUTPUT_PATH = os.path.join(VIZ_ROOT, "penalty_freekick_chefer_figure.png")

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE   = 448          # pixels per cell (square)
PAD         = 8            # gap between cells
COL_LABEL_H = 50           # header height
ROW_LABEL_W = 60           # left margin for row labels
BG_COLOR    = (255, 255, 255)
TEXT_COLOR  = (0, 0, 0)
LINE_COLOR  = (180, 180, 180)

COLUMNS  = ["Original", "MatchVision", "SigLIP", "SoccerMaster"]


ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(__file__), "annotations-coco.json"
)

# Category id → box colour (RGB)
CATEGORY_COLORS = {
    1: (150,   0, 255),   # small label  – Primary Cue
    2: (255,  20, 147),   # large label  – Secondary Cue
    3: (255, 182, 220),   # visual cue   – Common Cue
}


def load_annotations(video_id: str) -> dict:
    """Return {second: [annotation, ...]} for the given video_id."""
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    img_by_id = {img["id"]: img for img in data["images"]}
    result: dict = {}
    for ann in data["annotations"]:
        img = img_by_id.get(ann["image_id"])
        if img is None or img.get("video_id") != video_id:
            continue
        if ann.get("no_roi") or not ann.get("bbox") or ann["bbox"] == [0,0,0,0]:
            continue
        sec = img["second"]
        result.setdefault(sec, []).append(
            {"bbox": ann["bbox"], "category_id": ann["category_id"],
             "src_w": img["width"], "src_h": img["height"]}
        )
    return result


def draw_boxes(cell: Image.Image, anns: list) -> Image.Image:
    """Draw scaled COCO bboxes [x,y,w,h] onto cell — primary cues only."""
    cell = cell.copy()
    d = ImageDraw.Draw(cell)
    cw, ch = cell.size
    for ann in anns:
        if ann["category_id"] != 1:   # primary cue only
            continue
        x, y, bw, bh = ann["bbox"]
        sx = cw / ann["src_w"]
        sy = ch / ann["src_h"]
        x1, y1 = x * sx, y * sy
        x2, y2 = (x + bw) * sx, (y + bh) * sy
        color = CATEGORY_COLORS.get(ann["category_id"], (255, 0, 0))
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Return an RGB frame at the given 0-based frame index, or a grey placeholder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return np.full((CELL_SIZE, CELL_SIZE, 3), 200, dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  WARNING: frame {frame_idx} not found in {video_path}")
        return np.full((CELL_SIZE, CELL_SIZE, 3), 200, dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def to_square(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Resize the full frame to a square (no crop, no padding)."""
    return Image.fromarray(img_rgb).resize((size, size), Image.LANCZOS)


def crop_heatmap(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Take only the top `size` rows (the attribution overlay) and resize."""
    overlay = img_rgb[:size, :, :]          # 448 rows × 448 cols
    pil = Image.fromarray(overlay)
    return pil.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------
def build_figure():
    # Load annotations for this clip
    video_id = ORIGINAL_VIDEO.replace("\\", "/")
    video_id = video_id.replace(ORIGINAL_BASE.replace("\\", "/"), "").lstrip("/")
    ann_by_sec = load_annotations(video_id)

    n_rows = len(TIMESTAMPS)
    n_cols = len(COLUMNS)

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = COL_LABEL_H  + n_rows * CELL_SIZE + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    font   = get_font(26)

    # ---- Column headers ----
    for ci, label in enumerate(COLUMNS):
        x = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text((x, COL_LABEL_H // 2), label, fill=TEXT_COLOR,
                  font=font, anchor="mm")
        # underline
        xl = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        xr = xl + CELL_SIZE
        draw.line([(xl, COL_LABEL_H - 2), (xr, COL_LABEL_H - 2)],
                  fill=LINE_COLOR, width=2)

    # ---- Rows ----
    for ri, ts in enumerate(TIMESTAMPS):
        y_top = COL_LABEL_H + ri * (CELL_SIZE + PAD)

        # Row label (vertical text)
        label = f"t={ts}s"
        tmp      = Image.new("RGB", (CELL_SIZE, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((CELL_SIZE // 2, ROW_LABEL_W // 2), label,
                      fill=TEXT_COLOR, font=font, anchor="mm")
        rotated = tmp.rotate(90, expand=True)   # → (ROW_LABEL_W, CELL_SIZE)
        canvas.paste(rotated, (0, y_top))

        # ---- Original frame ----
        orig_frame_idx = ts * ORIGINAL_FPS
        print(f"  Original   t={ts}s  → frame {orig_frame_idx}")
        orig_img = read_frame(ORIGINAL_VIDEO, orig_frame_idx)
        cell = to_square(orig_img)
        if ts in ann_by_sec:
            cell = draw_boxes(cell, ann_by_sec[ts])
            print(f"    Drew {len(ann_by_sec[ts])} annotation(s) at t={ts}s")
        x = ROW_LABEL_W
        canvas.paste(cell, (x, y_top))

        # ---- Chefer model frames ----
        for ci, model in enumerate(COLUMNS[1:], start=1):
            vid_path = CHEFER_VIDEOS[model]
            # Chefer video is 1 fps, so frame index == timestamp in seconds
            chefer_frame_idx = ts
            print(f"  {model:14s} t={ts}s  → frame {chefer_frame_idx}  ({vid_path})")
            raw = read_frame(vid_path, chefer_frame_idx)
            cell = crop_heatmap(raw)
            x = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
            canvas.paste(cell, (x, y_top))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(300, 300))
    with open("/tmp/_throw_in_chefer_figure.png", "wb") as f:
        f.write(buf.getvalue())
    subprocess.run(["cp", "/tmp/_throw_in_chefer_figure.png", OUTPUT_PATH], check=True)
    print(f"\nSaved → {OUTPUT_PATH}  ({canvas.size[0]}×{canvas.size[1]} px)")


def build_multi_figure():
    """Build a figure with one row per ROW_ENTRIES clip."""
    n_rows = len(ROW_ENTRIES)
    n_cols = len(COLUMNS)

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = COL_LABEL_H  + n_rows * CELL_SIZE + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    font   = get_font(26)

    # ---- Column headers ----
    for ci, label in enumerate(COLUMNS):
        x = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text((x, COL_LABEL_H // 2), label, fill=TEXT_COLOR,
                  font=font, anchor="mm")
        xl = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        xr = xl + CELL_SIZE
        draw.line([(xl, COL_LABEL_H - 2), (xr, COL_LABEL_H - 2)],
                  fill=LINE_COLOR, width=2)

    # ---- Rows ----
    for ri, (row_label, video_rel, ts) in enumerate(ROW_ENTRIES):
        y_top = COL_LABEL_H + ri * (CELL_SIZE + PAD)

        # Row label (vertical text)
        tmp      = Image.new("RGB", (CELL_SIZE, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((CELL_SIZE // 2, ROW_LABEL_W // 2), row_label,
                      fill=TEXT_COLOR, font=font, anchor="mm")
        rotated = tmp.rotate(90, expand=True)
        canvas.paste(rotated, (0, y_top))

        # Parse paths
        parts     = video_rel.replace("\\", "/").split("/")
        league    = parts[0]
        match_dir = parts[1]
        clip_name = parts[2]

        # ---- Original frame ----
        orig_path = os.path.join(ORIGINAL_BASE, video_rel)
        cap = cv2.VideoCapture(orig_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        cap.release()
        orig_frame_idx = int(ts * orig_fps)
        print(f"  [{row_label}] original  t={ts}s → frame {orig_frame_idx}")
        orig_img = read_frame(orig_path, orig_frame_idx)
        cell = to_square(orig_img)
        # Draw primary cue annotations
        video_id = video_rel.replace("\\", "/")
        ann_by_sec = load_annotations(video_id)
        if ts in ann_by_sec:
            cell = draw_boxes(cell, ann_by_sec[ts])
            print(f"    Drew {len([a for a in ann_by_sec[ts] if a['category_id']==1])} primary cue(s) at t={ts}s")
        canvas.paste(cell, (ROW_LABEL_W, y_top))

        # ---- Chefer model frames ----
        for ci, model in enumerate(COLUMNS[1:], start=1):
            model_dir = {"MatchVision": "MatchVision", "SigLIP": "SigLip", "SoccerMaster": "SoccerMaster"}[model]
            vid_path  = os.path.join(VIZ_ROOT, model_dir, "Chefer", match_dir, clip_name)
            print(f"  [{row_label}] {model:14s} t={ts}s → frame {ts}  ({vid_path})")
            raw  = read_frame(vid_path, ts)
            cell = crop_heatmap(raw)
            canvas.paste(cell, (ROW_LABEL_W + ci * (CELL_SIZE + PAD), y_top))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(300, 300))
    with open("/tmp/_penalty_freekick_chefer_figure.png", "wb") as f:
        f.write(buf.getvalue())
    subprocess.run(["cp", "/tmp/_penalty_freekick_chefer_figure.png", MULTI_OUTPUT_PATH], check=True)
    print(f"\nSaved → {MULTI_OUTPUT_PATH}  ({canvas.size[0]}×{canvas.size[1]} px)")


if __name__ == "__main__":
    build_figure()
    build_multi_figure()
