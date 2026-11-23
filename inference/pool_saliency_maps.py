"""Average-pool saved saliency maps (T, H, W) down to a model's token grid.

`gradcam_inference.py` saves Grad-CAM maps upsampled to 224x224, while
`perturbation_faithfulness.py` needs one value per model token (14x14 for
MatchVision, 32x32 for SoccerMaster). This writes pooled copies and keeps the
`.json` side files unchanged.

    python inference/pool_saliency_maps.py <src_dir> <dst_dir> <grid>
"""
import glob
import os
import shutil
import sys

import numpy as np


def main(src, dst, grid):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(src, "*.npz"))):
        out = os.path.join(dst, os.path.basename(path))
        if os.path.exists(out):
            continue
        data = dict(np.load(path))
        maps = data["saliency_maps"].astype(np.float32)
        frames, height, width = maps.shape
        if height % grid or width % grid:
            raise ValueError(f"{path}: map {maps.shape} does not tile a {grid}x{grid} grid")
        data["saliency_maps_fullres_shape"] = np.array(maps.shape)
        data["saliency_maps"] = maps.reshape(frames, grid, height // grid, grid, width // grid).mean(axis=(2, 4))
        np.savez_compressed(out, **data)
        if os.path.exists(path[:-4] + ".json"):
            shutil.copy(path[:-4] + ".json", dst)
        n += 1
    print(f"pooled {n} maps -> {dst} (grid {grid})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
