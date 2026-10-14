#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.cm as cm


def load_depth_m_from_16bit_png(path: str) -> np.ndarray:
    """读入16-bit PNG（值=米×256），返回 float32 米制深度 [H,W]。"""
    arr = np.array(Image.open(path), dtype=np.uint16)
    depth_m = arr.astype(np.float32) / 256.0
    return depth_m


def colorize_depth_calibrated(
    depth_m: np.ndarray,
    min_m: float,
    max_m: float,
    cmap_name: str = "turbo",
    invalid_zero_black: bool = True,
) -> Image.Image:
    """
    固定量程上色：在[min_m,max_m]线性映射到[0,1]再走色图。
    depth==0 当作无效（可选）并画成黑色。
    """
    d = depth_m.copy()
    # 有效掩码
    mask_valid = d > 0 if invalid_zero_black else np.isfinite(d)

    # 裁剪并归一化
    d = np.clip(d, min_m, max_m).astype(np.float32)
    rng = max_m - min_m
    if rng <= 0:
        raise ValueError("`max-m` 必须大于 `min-m`。")
    norm = (d - min_m) / rng
    norm = np.clip(norm, 0.0, 1.0, dtype=np.float32)

    # 取色图
    cmap = cm.get_cmap(cmap_name)  # e.g., "turbo", "Spectral", "viridis"
    rgba = cmap(norm)              # [H,W,4], 0..1
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

    if invalid_zero_black:
        rgb[~mask_valid] = 0

    return Image.fromarray(rgb, mode="RGB")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def in_out_paths(in_path: str, in_root: str, out_root: str) -> Tuple[str, str]:
    """
    保持相对目录结构与文件名，只把扩展名改为 .png（彩色PNG）。
    """
    rel = os.path.relpath(in_path, start=in_root)
    rel_no_ext = os.path.splitext(rel)[0]
    out_path = os.path.join(out_root, rel_no_ext + ".png")
    ensure_dir(os.path.dirname(out_path))
    return in_path, out_path


def main():
    ap = argparse.ArgumentParser(
        description="将16-bit深度图（值=米×256）批量转换为标定彩色可视化PNG。"
    )
    ap.add_argument("--in-path", required=True,
                    help="输入：单个文件或目录（16-bit PNG）。")
    ap.add_argument("--out-dir", required=True,
                    help="输出彩色PNG的目录（保持相对结构与文件名）。")
    ap.add_argument("--min-m", type=float, default=0.0,
                    help="色标最小深度（米），默认0。")
    ap.add_argument("--max-m", type=float, default=80.0,
                    help="色标最大深度（米），默认80（KITTI常用）。")
    ap.add_argument("--cmap", type=str, default="turbo",
                    help="色图名称（turbo/Spectral/viridis/…），默认turbo。")
    ap.add_argument("--invalid-zero-black", action="store_true",
                    help="将深度=0的像素渲染为黑色（常用于稀疏/无效区域）。")
    args = ap.parse_args()

    in_path = args.in_path
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # 收集输入文件
    inputs = []
    if os.path.isdir(in_path):
        # 仅处理16-bit PNG
        pngs = glob(os.path.join(in_path, "**", "*.png"), recursive=True)
        inputs = [in_out_paths(p, in_path, out_dir) for p in pngs]
    elif os.path.isfile(in_path):
        # 单文件：输出到 out_dir 根目录，文件名不变、改为 .png
        base = os.path.basename(in_path)
        stem = os.path.splitext(base)[0]
        out_path = os.path.join(out_dir, stem + ".png")
        ensure_dir(out_dir)
        inputs = [(in_path, out_path)]
    else:
        raise FileNotFoundError(f"找不到输入路径：{in_path}")

    print(f"共发现 {len(inputs)} 个文件需要处理。")

    # 逐个处理
    for src, dst in inputs:
        depth_m = load_depth_m_from_16bit_png(src)
        vis = colorize_depth_calibrated(
            depth_m,
            min_m=args.min_m,
            max_m=args.max_m,
            cmap_name=args.cmap,
            invalid_zero_black=args.invalid_zero_black,
        )
        ensure_dir(os.path.dirname(dst))
        vis.save(dst)
        print(f"[OK] {src} -> {dst}")

    print("完成：已输出标定彩色深度PNG。")


if __name__ == "__main__":
    main()

