import os
from PIL import Image, ImageDraw, ImageFont

FRAMES_DIR = "extracted_frames_chefer_compare"
OUTPUT_PATH = "paper_figure_chefer_compare.png"

CLASSES = [
    "red_card",
    "yellow_card",
    "corner",
    "penalty",
    "foul_with_no_card",
    "injury",
    "lead_to_corner",
    "throw_in",
    "substitution",
    "free_kick",
    "goal",
    "foul_lead_to_penalty",
    "second_yellow_card",
]

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

# Column order: (method_tag, model)
COLUMNS = [
    ("chefer-spatial",  "MatchVision"),
    ("chefer-temporal", "MatchVision"),
    ("chefer-spatial",  "SoccerMaster"),
    ("chefer-temporal", "SoccerMaster"),
]

COL_LABELS = ["Spatial", "Spatial+Temporal", "Spatial", "Spatial+Temporal"]

MODEL_GROUPS = [
    ("MatchVision",  0, 2),
    ("SoccerMaster", 2, 2),
]

CELL_SIZE = 448
PAD = 4
ROW_LABEL_W = 280
MODEL_LABEL_H = 50
COL_LABEL_H = 50
HEADER_H = MODEL_LABEL_H + COL_LABEL_H
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
LINE_COLOR = (180, 180, 180)


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
    total_h = HEADER_H + n_rows * CELL_SIZE + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    model_font = get_font(32)
    header_font = get_font(26)
    row_font = get_font(26)

    for label, start_col, span in MODEL_GROUPS:
        x_left = ROW_LABEL_W + start_col * (CELL_SIZE + PAD)
        x_right = x_left + span * CELL_SIZE + (span - 1) * PAD
        x_center = (x_left + x_right) // 2
        draw.text(
            (x_center, MODEL_LABEL_H // 2), label, fill=TEXT_COLOR,
            font=model_font, anchor="mm",
        )
        draw.line(
            [(x_left, MODEL_LABEL_H - 2), (x_right, MODEL_LABEL_H - 2)],
            fill=LINE_COLOR, width=2,
        )

    for ci, label in enumerate(COL_LABELS):
        x_center = ROW_LABEL_W + ci * (CELL_SIZE + PAD) + CELL_SIZE // 2
        draw.text(
            (x_center, MODEL_LABEL_H + COL_LABEL_H // 2), label,
            fill=TEXT_COLOR, font=header_font, anchor="mm",
        )

    for ri, cls in enumerate(classes):
        y_top = HEADER_H + ri * (CELL_SIZE + PAD)

        label = CLASS_LABELS.get(cls, cls)
        y_center = y_top + CELL_SIZE // 2
        draw.text(
            (ROW_LABEL_W - 16, y_center), label, fill=TEXT_COLOR,
            font=row_font, anchor="rm",
        )

        for ci, (method, model) in enumerate(COLUMNS):
            fname = f"{cls}_{method}_{model}.png"
            fpath = os.path.join(FRAMES_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath}")
                continue
            img = Image.open(fpath)
            x = ROW_LABEL_W + ci * (CELL_SIZE + PAD)
            canvas.paste(img, (x, y_top))

    canvas.save(output_path, dpi=(300, 300))
    print(f"Saved {output_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    build_figure(CLASSES, OUTPUT_PATH)


if __name__ == "__main__":
    main()
