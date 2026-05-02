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
    ("gradcam", "MatchVision"),
    ("chefer", "SigLip"),
    ("gradcam", "SigLip"),
    ("chefer", "SoccerMaster"),
    ("gradcam", "SoccerMaster"),
]

COL_LABELS = ["", "Chefer", "Grad-CAM", "Chefer", "Grad-CAM", "Chefer", "Grad-CAM"]

# Model group headers spanning 2 columns each
# start_col indices are relative to the full COLUMNS list
MODEL_GROUPS = [
    ("Original", 0, 1),      # label, start_col, span
    ("MatchVision", 1, 2),
    ("SigLIP", 3, 2),
    ("SoccerMaster", 5, 2),
]

CELL_SIZE = 448  # native resolution per image
PAD = 4  # padding between cells
ROW_LABEL_W = 80   # width reserved for row labels (vertical text)
MODEL_LABEL_H = 50  # height for top model group header
COL_LABEL_H = 50   # height for sub-column method labels
HEADER_H = MODEL_LABEL_H + COL_LABEL_H  # total header height
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
    n_rows = len(classes)
    n_cols = len(COLUMNS)

    total_w = ROW_LABEL_W + n_cols * CELL_SIZE + (n_cols - 1) * PAD
    total_h = HEADER_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD + LEGEND_H

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    model_font = get_font(32)
    header_font = get_font(26)
    row_font = get_font(26)

    # Draw model group headers (top row)
    for label, start_col, span in MODEL_GROUPS:
        x_left = ROW_LABEL_W + start_col * (CELL_SIZE + PAD)
        x_right = x_left + span * CELL_SIZE + (span - 1) * PAD
        x_center = (x_left + x_right) // 2
        draw.text(
            (x_center, MODEL_LABEL_H // 2), label, fill=TEXT_COLOR,
            font=model_font, anchor="mm",
        )
        # Draw underline for the group
        draw.line(
            [(x_left, MODEL_LABEL_H - 2), (x_right, MODEL_LABEL_H - 2)],
            fill=LINE_COLOR, width=2,
        )

    # Draw sub-column method labels
    for ci, label in enumerate(COL_LABELS):
        x_center = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text(
            (x_center, MODEL_LABEL_H + COL_LABEL_H // 2), label,
            fill=TEXT_COLOR, font=header_font, anchor="mm",
        )

    # Draw rows
    for ri, cls in enumerate(classes):
        y_top = HEADER_H + ri * (CELL_SIZE + PAD)

        # Row label — drawn vertically
        label = CLASS_LABELS.get(cls, cls)
        y_center = y_top + CELL_SIZE // 2
        # Render text onto a temporary image then rotate 90°
        tmp = Image.new("RGB", (CELL_SIZE, ROW_LABEL_W), BG_COLOR)
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text(
            (CELL_SIZE // 2, ROW_LABEL_W // 2), label, fill=TEXT_COLOR,
            font=row_font, anchor="mm",
        )
        rotated = tmp.rotate(90, expand=True)  # becomes (ROW_LABEL_W, CELL_SIZE)
        canvas.paste(rotated, (0, y_top))

        # Paste images
        for ci, (method, model) in enumerate(COLUMNS):
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
    legend_y = HEADER_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD
    legend_font = get_font(24)
    swatch_size = 24
    swatch_pad = 10
    # Centre the legend items horizontally across the full canvas
    item_w = swatch_size + swatch_pad + 400  # approx width per item (wider spacing)
    total_legend_w = len(LEGEND_ITEMS) * item_w
    legend_x = (total_w - total_legend_w) // 2
    legend_cy = legend_y + LEGEND_H // 2
    for label, rgb in LEGEND_ITEMS:
        # Coloured square swatch
        sx, sy = legend_x, legend_cy - swatch_size // 2
        draw.rectangle([sx, sy, sx + swatch_size, sy + swatch_size], fill=rgb)
        # Label to the right of the swatch
        draw.text(
            (sx + swatch_size + swatch_pad, legend_cy),
            label, fill=TEXT_COLOR, font=legend_font, anchor="lm",
        )
        legend_x += item_w

    canvas.save(output_path, dpi=(300, 300))
    print(f"Saved {output_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    for split in FIGURE_SPLITS:
        build_figure(split["classes"], split["output"])


if __name__ == "__main__":
    main()
