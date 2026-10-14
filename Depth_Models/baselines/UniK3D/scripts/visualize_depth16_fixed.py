import os
import glob
import argparse

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm as cm


def depth16_x256_to_meters(depth_u16: np.ndarray) -> np.ndarray:
    return depth_u16.astype(np.float32) / 256.0


def colorize_fixed(depth_m: np.ndarray, vmin: float, vmax: float, cmap_name: str = "magma") -> np.ndarray:
    """
    固定映射：depth(m) -> color
    - depth<=0 视为无效，输出黑色
    - clip 到 [vmin, vmax]
    """
    depth = depth_m.copy()
    valid = depth > 0

    depth = np.clip(depth, vmin, vmax)
    norm = (depth - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)

    # invalid -> black
    rgb[~valid] = 0
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="input depth16 png or a folder")
    parser.add_argument("--out_dir", required=True, help="output folder for colored images")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=80.0)
    parser.add_argument("--cmap", default="magma", help="matplotlib colormap name, e.g. magma/viridis/plasma/inferno")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.isdir(args.in_path):
        depth_paths = sorted(glob.glob(os.path.join(args.in_path, "*.png")))
    else:
        depth_paths = [args.in_path]

    if len(depth_paths) == 0:
        raise RuntimeError(f"No png found in {args.in_path}")

    for p in depth_paths:
        d_u16 = np.array(Image.open(p))
        if d_u16.dtype != np.uint16:
            # 有些 PIL 会读成 int32/uint32，这里做一次安全转换
            d_u16 = d_u16.astype(np.uint16)

        d_m = depth16_x256_to_meters(d_u16)
        rgb = colorize_fixed(d_m, vmin=args.vmin, vmax=args.vmax, cmap_name=args.cmap)

        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_dir, stem + f"_fixed_{args.cmap}_{args.vmin:g}-{args.vmax:g}m.png")
        Image.fromarray(rgb).save(out_path)

    print("Done. Saved to:", args.out_dir)


if __name__ == "__main__":
    main()


