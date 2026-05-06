"""
Two-row figure driven by one_video_for_visualization.json.

Layout:
    Rows    : Original | SoccerMaster Chefer
    Columns : one per JSON entry, labelled with the class name

Run:
    python make_class_figure.py
"""

import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JSON_PATH = "train_data/json/one_video_for_visualization.json"

VIZ_ROOT = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLensVisualizations-2-5-2026-chefer-full/SoccerLensVisualizations-2-5-2026"
)
ORIGINAL_BASE = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLens-for-annotation/SoccerLens-for-annotation"
)

OUTPUT_PATH = os.path.join(VIZ_ROOT, "class_chefer_figure.png")

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE   = 448
PAD         = 8
COL_LABEL_H = 60    # height for class name header above columns
ROW_LABEL_W = 120   # width for row labels on the left
BG_COLOR    = (255, 255, 255)
TEXT_COLOR  = (0, 0, 0)
LINE_COLOR  = (180, 180, 180)

ROWS = ["Original", "SoccerMaster\nChefer"]

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
    """Return an RGB frame at the given 0-based index, or a grey placeholder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return np.full((CELL_SIZE, CELL_SIZE, 3), 200, dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ret:
        print(f"  WARNING: frame {frame_idx} not found in {video_path}")
        return np.full((CELL_SIZE, CELL_SIZE, 3), 200, dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps


def read_frame_simple(video_path: str, frame_idx: int) -> np.ndarray:
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


def to_pil(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Resize (no crop) to square cell."""
    pil = Image.fromarray(img_rgb)
    return pil.resize((size, size), Image.LANCZOS)


def crop_heatmap(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Take only the top `size` rows (attribution overlay) and resize."""
    overlay = img_rgb[:size, :, :]
    pil = Image.fromarray(overlay)
    return pil.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_figure():
    with open(JSON_PATH) as f:
        entries = json.load(f)

    n_cols = len(entries)
    n_rows = 2  # Original + SoccerMaster Chefer

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = COL_LABEL_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    col_font = get_font(28)
    row_font = get_font(24)

    # ---- Column headers (class names) ----
    for ci, entry in enumerate(entries):
        label = entry["caption"].replace("_", " ").title()
        x = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text((x, COL_LABEL_H // 2), label, fill=TEXT_COLOR,
                  font=col_font, anchor="mm")
        xl = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        xr = xl + CELL_SIZE
        draw.line([(xl, COL_LABEL_H - 2), (xr, COL_LABEL_H - 2)],
                  fill=LINE_COLOR, width=2)

    # ---- Row labels ----
    for ri, row_label in enumerate(ROWS):
        y_top = COL_LABEL_H + ri * (CELL_SIZE + PAD)
        y_center = y_top + CELL_SIZE // 2
        # Render onto a temp image then rotate
        lines = row_label.split("\n")
        line_h = 30
        tmp_h = max(CELL_SIZE, len(lines) * (line_h + 4))
        tmp = Image.new("RGB", (tmp_h, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        for li, line in enumerate(lines):
            y_off = tmp_h // 2 + (li - (len(lines) - 1) / 2) * (line_h + 4)
            tmp_draw.text((tmp_h // 2, int(y_off)), line,
                          fill=TEXT_COLOR, font=row_font, anchor="mm")
        rotated = tmp.rotate(90, expand=True)  # → (ROW_LABEL_W, tmp_h)
        canvas.paste(rotated.crop((0, 0, ROW_LABEL_W, CELL_SIZE)), (0, y_top))

    # ---- Fill cells ----
    for ci, entry in enumerate(entries):
        video_rel  = entry["video"].replace("\\", "/")
        ts         = entry["timestamp"]
        parts      = video_rel.split("/")
        league     = parts[0]
        match_dir  = parts[1]
        clip_name  = parts[2]

        # --- Row 0: Original ---
        orig_path = os.path.join(ORIGINAL_BASE, video_rel)
        cap = cv2.VideoCapture(orig_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        cap.release()
        orig_frame_idx = int(ts * orig_fps)
        print(f"  [{entry['caption']}] original  t={ts}s → frame {orig_frame_idx}  ({orig_path})")
        orig_img = read_frame_simple(orig_path, orig_frame_idx)
        cell = to_pil(orig_img)
        x = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        canvas.paste(cell, (x, COL_LABEL_H))

        # --- Row 1: SoccerMaster Chefer ---
        chefer_path = os.path.join(VIZ_ROOT, "SoccerMaster", "Chefer", match_dir, clip_name)
        print(f"  [{entry['caption']}] chefer     t={ts}s → frame {ts}  ({chefer_path})")
        raw = read_frame_simple(chefer_path, ts)
        cell = crop_heatmap(raw)
        y_top = COL_LABEL_H + CELL_SIZE + PAD
        canvas.paste(cell, (x, y_top))

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"\nSaved → {OUTPUT_PATH}  ({canvas.size[0]}×{canvas.size[1]} px)")


if __name__ == "__main__":
    build_figure()
