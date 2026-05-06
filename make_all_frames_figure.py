"""
Build a figure showing all frames of one clip: heatmaps vs original.

Layout:
    Columns : Original | MatchVision | SigLIP | SoccerMaster
    Rows    : one per frame (timestamp) in the Chefer video

Output saved to OUTPUT_PATH (Linux filesystem by default).

Run:
    python make_all_frames_figure.py
"""

import io
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIZ_ROOT = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLensVisualizations-2-5-2026-chefer-full/SoccerLensVisualizations-2-5-2026"
)
ORIGINAL_BASE = (
    r"/mnt/c/Users/z004kjmt/Downloads"
    r"/SoccerLens-for-annotation/SoccerLens-for-annotation"
)

# The clip to visualize (relative to ORIGINAL_BASE / model Chefer dirs)
VIDEO_REL = "germany_bundesliga_2015-2016/2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen/2_14_30.mp4"

# Save to Linux filesystem to avoid full Windows C: drive
OUTPUT_PATH = "/tmp/bayern_all_frames_soccermaster.png"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE   = 224          # smaller cells so the figure isn't enormous
PAD         = 4
COL_LABEL_H = 40
ROW_LABEL_W = 50
BG_COLOR    = (255, 255, 255)
TEXT_COLOR  = (0, 0, 0)
LINE_COLOR  = (180, 180, 180)

COLUMNS = ["Original", "SoccerMaster"]

MODEL_DIRS = {
    "MatchVision":  "MatchVision",
    "SigLIP":       "SigLip",
    "SoccerMaster": "SoccerMaster",
}

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


def to_cell(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Stretch-resize to square cell (no crop, no padding)."""
    return Image.fromarray(img_rgb).resize((size, size), Image.LANCZOS)


def crop_heatmap(img_rgb: np.ndarray, size: int = CELL_SIZE) -> Image.Image:
    """Take top `size` rows (attribution overlay) and resize to cell."""
    overlay = img_rgb[:size, :, :]
    return Image.fromarray(overlay).resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build():
    parts     = VIDEO_REL.replace("\\", "/").split("/")
    match_dir = parts[1]
    clip_name = parts[2]

    orig_path = os.path.join(ORIGINAL_BASE, VIDEO_REL)

    chefer_paths = {
        model: os.path.join(VIZ_ROOT, MODEL_DIRS[model], "Chefer", match_dir, clip_name)
        for model in COLUMNS[1:]
    }

    # Detect frame count from one Chefer video
    cap = cv2.VideoCapture(chefer_paths[COLUMNS[1]])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {chefer_paths['MatchVision']}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Clip: {VIDEO_REL}")
    print(f"Total frames: {n_frames}")

    # Get original FPS
    cap = cv2.VideoCapture(orig_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()
    print(f"Original FPS: {orig_fps}")

    n_rows = n_frames
    n_cols = len(COLUMNS)

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = COL_LABEL_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD

    print(f"Canvas size: {total_w} × {total_h} px")

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    font   = get_font(16)

    # Column headers
    for ci, label in enumerate(COLUMNS):
        x = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text((x, COL_LABEL_H // 2), label, fill=TEXT_COLOR, font=font, anchor="mm")
        xl = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        xr = xl + CELL_SIZE
        draw.line([(xl, COL_LABEL_H - 2), (xr, COL_LABEL_H - 2)], fill=LINE_COLOR, width=2)

    for ts in range(n_frames):
        y_top = COL_LABEL_H + ts * (CELL_SIZE + PAD)

        # Row label
        label = f"t={ts}s"
        tmp      = Image.new("RGB", (CELL_SIZE, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((CELL_SIZE // 2, ROW_LABEL_W // 2), label,
                      fill=TEXT_COLOR, font=font, anchor="mm")
        rotated = tmp.rotate(90, expand=True)
        canvas.paste(rotated, (0, y_top))

        # Original
        orig_frame_idx = int(ts * orig_fps)
        print(f"  t={ts:2d}s  orig→frame {orig_frame_idx}", end="")
        orig_img = read_frame(orig_path, orig_frame_idx)
        canvas.paste(to_cell(orig_img), (ROW_LABEL_W, y_top))

        # Models
        for ci, model in enumerate(COLUMNS[1:], start=1):
            raw  = read_frame(chefer_paths[model], ts)
            cell = crop_heatmap(raw)
            canvas.paste(cell, (ROW_LABEL_W + ci * (CELL_SIZE + PAD), y_top))
            print(f"  {model}✓", end="")
        print()

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(150, 150))
    with open(OUTPUT_PATH, "wb") as f:
        f.write(buf.getvalue())
    print(f"\nSaved → {OUTPUT_PATH}  ({canvas.size[0]}×{canvas.size[1]} px)")


if __name__ == "__main__":
    build()
