#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mean_std(img_paths, max_images=None):
    """
    计算 RGB mean/std（像素量纲 0..255），使用在线累积避免一次性读入全部像素。
    返回:
        mean: (3,)
        std:  (3,)
    """
    if max_images is not None:
        img_paths = img_paths[:max_images]

    # 累积 sum / sumsq / count
    sum_rgb = np.zeros(3, dtype=np.float64)
    sumsq_rgb = np.zeros(3, dtype=np.float64)
    count = 0  # 总像素数 * 通道按每通道独立统计

    for p in tqdm(img_paths, desc="Computing mean/std", dynamic_ncols=True):
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)  # HxWx3, 0..255

        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3)  # (H*W, 3)

        sum_rgb += pixels.sum(axis=0)
        sumsq_rgb += (pixels ** 2).sum(axis=0)
        count += pixels.shape[0]

    mean = sum_rgb / max(count, 1)
    var = sumsq_rgb / max(count, 1) - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return mean, std, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True, help="DENSE image folder, e.g. .../val_selection_cropped_DENSE/image")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern, default: *.png")
    parser.add_argument("--max-images", type=int, default=None, help="Use only first N images (for quick check)")
    args = parser.parse_args()

    img_dir = args.img_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img-dir not found: {img_dir}")

    img_paths = sorted(glob(os.path.join(img_dir, args.pattern)))
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir} with pattern {args.pattern}")

    mean, std, count = compute_mean_std(img_paths, max_images=args.max_images)

    print("\n================== RESULT ==================")
    print(f"Images used: {len(img_paths) if args.max_images is None else min(len(img_paths), args.max_images)}")
    print(f"Total pixels: {count}")
    print("Mean (RGB, 0..255):", mean.tolist())
    print("Std  (RGB, 0..255):", std.tolist())

    # YAML snippet (matching your config style)
    print("\nPaste into configs/data/DENSE.yaml:")
    print("mean:")
    print(f"  - {mean[0]:.4f}")
    print(f"  - {mean[1]:.4f}")
    print(f"  - {mean[2]:.4f}")
    print("std:")
    print(f"  - {std[0]:.4f}")
    print(f"  - {std[1]:.4f}")
    print(f"  - {std[2]:.4f}")
    print("============================================\n")


if __name__ == "__main__":
    main()


# python compute_mean_std_dense.py \
#   --img-dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC/val_selection_cropped_DC/image
