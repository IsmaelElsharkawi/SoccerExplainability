"""Show what the model sees at each masked fraction of the perturbation test.

Renders a montage for one clip: for MoRF (most relevant first) and LeRF
(least relevant first), the selected frames after cumulatively masking 10%,
20%, ... of the clip's tokens, using the same `rank_groups` as
`perturbation_faithfulness.py`. Masked patches are filled with the clip's mean
colour, which is what ``--fill mean`` does in the model's normalized space.
Each tile is annotated with the share of *that frame* that is masked; with
``--ranking clip`` this differs from the clip-level fraction, since frames
with a high temporal weight are masked first.

    python inference/visualize_perturbation_masks.py \
        --npz <run>/soccermaster/saliency/<clip>.npz \
        --video /path/to/<clip>.mp4 --frames 0 14 29 --out masks.png

``--frames_dir`` with per-frame PNG/JPG files (sorted) works instead of
``--video``. A saliency ``.npy`` of shape (T, h, w) can replace ``--npz``.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perturbation_faithfulness import rank_groups  # noqa: E402

LABEL_HEIGHT = 22
MARGIN = 6


def load_frames(args, num_frames):
    """uint8 (T, H, W, 3) frames of the clip, sampled like the attribution runs."""
    if args.frames_dir:
        names = sorted(n for n in os.listdir(args.frames_dir) if n.lower().endswith((".png", ".jpg", ".jpeg")))
        frames = np.stack([np.asarray(Image.open(os.path.join(args.frames_dir, n)).convert("RGB")) for n in names])
    else:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from dataset.video_utils_siglip import read_frames_decord
        frames, _, _ = read_frames_decord(args.video, num_frames, args.sample)
        frames = frames.permute(0, 2, 3, 1).numpy()
    if frames.shape[0] != num_frames:
        raise ValueError(f"{frames.shape[0]} frames but the saliency volume has {num_frames}")
    return frames


def load_saliency(path, map_key):
    if path.endswith(".npy"):
        return np.load(path)
    with np.load(path) as data:
        return data[map_key]


def apply_mask(frame, token_mask, fill_rgb):
    """Grey out masked tokens on one (H, W, 3) frame; tokens tile the frame."""
    height, width = frame.shape[:2]
    grid_h, grid_w = token_mask.shape
    mask = np.repeat(np.repeat(token_mask, int(np.ceil(height / grid_h)), axis=0),
                     int(np.ceil(width / grid_w)), axis=1)[:height, :width]
    out = frame.copy()
    out[mask] = fill_rgb
    return out


def cumulative_masks(groups, num_groups, order):
    """Yield (fraction, token_mask) for the cumulative steps of one order."""
    for i in range(1, num_groups):
        if order == "morf":
            yield i / num_groups, groups < i
        else:
            yield i / num_groups, groups >= num_groups - i


def render(frames, saliency, frame_ids, num_groups, ranking, tile_width, title):
    groups = rank_groups(saliency, num_groups, ranking)
    fill_rgb = frames.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    height, width = frames.shape[1:3]
    tile_height = int(round(height * tile_width / width))
    steps = list(range(1, num_groups))
    columns = 1 + len(steps)
    rows = 2 * len(frame_ids)
    canvas_w = MARGIN + columns * (tile_width + MARGIN)
    canvas_h = LABEL_HEIGHT * 2 + MARGIN + rows * (tile_height + LABEL_HEIGHT + MARGIN) + LABEL_HEIGHT
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 4), title, fill=(0, 0, 0))
    for c, frac in enumerate([0.0] + [i / num_groups for i in steps]):
        x = MARGIN + c * (tile_width + MARGIN)
        draw.text((x, LABEL_HEIGHT + 4), "original" if c == 0 else f"{frac:.0%} of clip masked", fill=(0, 0, 0))

    y = 2 * LABEL_HEIGHT + MARGIN
    for order in ("morf", "lerf"):
        masks = [(0.0, np.zeros(saliency.shape, dtype=bool))] + list(cumulative_masks(groups, num_groups, order))
        for t in frame_ids:
            name = "MoRF" if order == "morf" else "LeRF"
            draw.text((MARGIN, y), f"{name}  frame {t}  (frame peak salience {saliency[t].max():.2f})", fill=(0, 0, 0))
            y += LABEL_HEIGHT
            for c, (_, token_mask) in enumerate(masks):
                tile = apply_mask(frames[t], token_mask[t], fill_rgb)
                tile = Image.fromarray(tile).resize((tile_width, tile_height), Image.BILINEAR)
                x = MARGIN + c * (tile_width + MARGIN)
                canvas.paste(tile, (x, y))
                share = token_mask[t].mean()
                ImageDraw.Draw(canvas).rectangle([x, y, x + 118, y + 14], fill=(0, 0, 0))
                ImageDraw.Draw(canvas).text((x + 3, y + 1), f"this frame: {share:.0%} masked", fill=(255, 255, 255))
            y += tile_height + MARGIN
    draw.text((MARGIN, canvas_h - LABEL_HEIGHT + 4),
              f"K={num_groups} groups, ranking={ranking}, masked tokens filled with the clip mean colour "
              f"(what --fill mean shows the model). MoRF removes the most salient tokens first, LeRF the least.",
              fill=(0, 0, 0))
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", required=True, help="Saved saliency .npz (or a (T, h, w) .npy).")
    parser.add_argument("--map_key", default="saliency_maps")
    parser.add_argument("--video", default=None, help="Clip to decode (same sampling as the attribution runs).")
    parser.add_argument("--frames_dir", default=None, help="Directory of per-frame images instead of --video.")
    parser.add_argument("--sample", default="middle")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="Frame indices to show (default: lowest, median and highest frame peak salience).")
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--ranking", choices=("clip", "frame"), default="clip")
    parser.add_argument("--tile_width", type=int, default=160)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True, help="Output PNG.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video and not args.frames_dir:
        raise SystemExit("one of --video / --frames_dir is required")
    saliency = load_saliency(args.npz, args.map_key).astype(np.float64)
    frames = load_frames(args, saliency.shape[0])
    if args.frames is None:
        peaks = saliency.reshape(saliency.shape[0], -1).max(axis=1)
        order = np.argsort(peaks)
        args.frames = sorted({int(order[0]), int(order[len(order) // 2]), int(order[-1])})
    title = args.title or os.path.basename(args.npz)
    canvas = render(frames, saliency, args.frames, args.groups, args.ranking, args.tile_width, title)
    canvas.save(args.out)
    print(f"wrote {args.out} ({canvas.size[0]}x{canvas.size[1]}), frames {args.frames}")


if __name__ == "__main__":
    main()
