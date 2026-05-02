"""
Build a figure showing Chefer attributions at specific timestamps.

Layout:
    Columns : Original | MatchVision | SigLIP | SoccerMaster
    Rows    : one per timestamp

Run:
    python make_temporal_figure.py"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIZ_ROOT = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLensVisualizations-2-5-2026-v5/SoccerLensVisualizations-2-5-2026"
)
MATCH_DIR = "2015-04-25 - 17-00 Espanyol 0 - 2 Barcelona"
CHEFER_VIDEO_NAME = "2_20_15.mp4"

ORIGINAL_VIDEO = (
    r"/mnt/c/Users/z004kjmt/Downloads/SoccerLens-for-annotation"
    r"/SoccerLens-for-annotation/spain_laliga_2014-2015"
    r"/2015-04-25 - 17-00 Espanyol 0 - 2 Barcelona/2_20_15.mp4"
)
ORIGINAL_FPS = 30  # original clip fps

CHEFER_VIDEOS = {
    "MatchVision":  os.path.join(VIZ_ROOT, "MatchVision",   "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
    "SigLIP":       os.path.join(VIZ_ROOT, "SigLip",        "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
    "SoccerMaster": os.path.join(VIZ_ROOT, "SoccerMaster",  "Chefer", MATCH_DIR, CHEFER_VIDEO_NAME),
}

TIMESTAMPS = [15]   # seconds into the clip

OUTPUT_PATH = os.path.join(VIZ_ROOT, "temporal_chefer_figure.png")

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
    """Center-crop to square then resize to `size`."""
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    cropped = img_rgb[top:top + side, left:left + side]
    pil = Image.fromarray(cropped)
    return pil.resize((size, size), Image.LANCZOS)


def crop_heatmap(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Take only the top `size` rows (the attribution overlay) and resize."""
    overlay = img_rgb[:size, :, :]          # 448 rows × 448 cols
    pil = Image.fromarray(overlay)
    return pil.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------
def build_figure():
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

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"\nSaved → {OUTPUT_PATH}  ({canvas.size[0]}×{canvas.size[1]} px)")


if __name__ == "__main__":
    build_figure()
