import os
from PIL import Image, ImageDraw, ImageFont

FRAMES_DIR = "extracted_frames"

# Split into two figures
FIGURE_SPLITS = [
    {
        "output": "paper_figure_part1.png",
        "classes": [
            "injury",
            "red_card",
            "yellow_card",
            "corner",
            "substitution",
            "lead_to_corner",
            "second_yellow_card",
        ],
    },
    {
        "output": "paper_figure_part2.png",
        "classes": [
            "penalty",
            "foul_with_no_card",
            "throw_in",
            "free_kick",
            "goal",
            "foul_lead_to_penalty",
        ],
    },
]

# Display labels for rows
CLASS_LABELS = {
    "red_card": "Red Card",
    "yellow_card": "Yellow Card",
    "corner": "Corner",
    "penalty": "Penalty",
    "foul_with_no_card": "Foul (No Card)",
    "injury": "Injury",
    "lead_to_corner": "Lead to Corner",
    "throw_in": "Throw In",
    "substitution": "Substitution",
    "free_kick": "Free Kick",
    "goal": "Goal",
    "foul_lead_to_penalty": "Lead to Penalty",
    "second_yellow_card": "2nd Yellow Card",
}

# Column order: (method, model) — None means original
COLUMNS = [
    (None, None),            # Original
    ("chefer", "MatchVision"),
    ("chefer", "SigLip"),
    ("chefer", "SoccerMaster"),
]

COL_LABELS = ["Original", "MatchVision", "SigLIP", "SoccerMaster"]

# No model group headers needed — single row of column labels
MODEL_GROUPS = []

CELL_SIZE = 448  # native resolution per image
PAD = 6  # padding between cells
ROW_LABEL_W = 110  # width reserved for row labels (model names)
COL_LABEL_H = 55   # height for class column headers
LEGEND_H = 70  # height for the bbox color legend at the bottom
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
LINE_COLOR = (180, 180, 180)

# BGR colors from extract_frames.py converted to RGB for PIL
LEGEND_ITEMS = [
    ("Primary Cue",   (150,   0, 255)),   # BGR(255,   0, 150) → RGB
    ("Secondary Cue", (255,  20, 147)),   # BGR(147,  20, 255) → RGB
    ("Common Cue",    (255, 182, 220)),   # light pink RGB
]


def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_figure(classes, output_path):
    """
    Horizontal layout: rows = models (Original, MatchVision, SigLIP, SoccerMaster)
                       cols = event classes
    """
    n_cols = len(classes)   # e.g. 6 or 7
    n_rows = len(COLUMNS)   # 4

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = COL_LABEL_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD + LEGEND_H

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(26)
    row_font    = get_font(26)

    # ---- Column headers (class names) ----
    for ci, cls in enumerate(classes):
        label = CLASS_LABELS.get(cls, cls)
        x_center = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text((x_center, COL_LABEL_H // 2), label,
                  fill=TEXT_COLOR, font=header_font, anchor="mm")
        xl = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
        xr = xl + CELL_SIZE
        draw.line([(xl, COL_LABEL_H - 2), (xr, COL_LABEL_H - 2)],
                  fill=LINE_COLOR, width=2)

    # ---- Rows (models) ----
    for ri, (method, model) in enumerate(COLUMNS):
        y_top = COL_LABEL_H + ri * (CELL_SIZE + PAD)

        # Row label (vertical text in the left margin)
        row_label = COL_LABELS[ri]
        tmp = Image.new("RGB", (CELL_SIZE, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((CELL_SIZE // 2, ROW_LABEL_W // 2), row_label,
                      fill=TEXT_COLOR, font=row_font, anchor="mm")
        rotated = tmp.rotate(90, expand=True)  # (ROW_LABEL_W, CELL_SIZE)
        canvas.paste(rotated, (0, y_top))
        draw.line([(0, y_top), (ROW_LABEL_W - 4, y_top)],
                  fill=LINE_COLOR, width=1)

        # ---- Cells ----
        for ci, cls in enumerate(classes):
            if method is None:
                fname = f"{cls}_original.png"
            else:
                fname = f"{cls}_{method}_{model}.png"
            fpath = os.path.join(FRAMES_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath}")
                continue
            img = Image.open(fpath).convert("RGB")
            img = img.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
            x = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
            canvas.paste(img, (x, y_top))

    # ---- Legend ----
    legend_y  = COL_LABEL_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD
    legend_font = get_font(24)
    swatch_size = 24
    swatch_pad  = 10
    item_w = swatch_size + swatch_pad + 400
    total_legend_w = len(LEGEND_ITEMS) * item_w
    legend_x  = (total_w - total_legend_w) // 2
    legend_cy = legend_y + LEGEND_H // 2
    for label, rgb in LEGEND_ITEMS:
        sx, sy = legend_x, legend_cy - swatch_size // 2
        draw.rectangle([sx, sy, sx + swatch_size, sy + swatch_size], fill=rgb)
        draw.text((sx + swatch_size + swatch_pad, legend_cy),
                  label, fill=TEXT_COLOR, font=legend_font, anchor="lm")
        legend_x += item_w

    canvas.save(output_path, dpi=(300, 300))
    print(f"Saved {output_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    for split in FIGURE_SPLITS:
        build_figure(split["classes"], split["output"])


if __name__ == "__main__":
    main()
