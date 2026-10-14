#!/usr/bin/env python3
import os
import argparse
import numpy as np

def read_kitti_bin_points(bin_file_path: str) -> np.ndarray:
    arr = np.fromfile(bin_file_path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid .bin size for {bin_file_path}")
    pts = arr.reshape(-1, 4)
    # 过滤 NaN/Inf
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

def iter_bin_files(root: str, recursive: bool):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".bin"):
                    yield os.path.join(dirpath, f)
    else:
        for f in sorted(os.listdir(root)):
            if f.endswith(".bin"):
                yield os.path.join(root, f)

def main():
    ap = argparse.ArgumentParser(description="Scan .bin point clouds and report the maximum height (max z).")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing .bin files")
    ap.add_argument("--recursive", action="store_true", help="Recursively search subfolders")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    global_max = None
    max_file = None
    num_files = 0
    num_bad = 0

    for fp in iter_bin_files(args.folder, args.recursive):
        try:
            pts = read_kitti_bin_points(fp)
        except Exception as e:
            print(f"[Skip] {fp}: {e}")
            num_bad += 1
            continue
        if pts.shape[0] == 0:
            continue
        zmax = float(pts[:, 2].max())
        num_files += 1
        if (global_max is None) or (zmax > global_max):
            global_max = zmax
            max_file = fp

    if num_files == 0:
        print("❗未找到可用的 .bin 文件（可能目录为空或全部损坏）。")
        return

    if global_max is None:
        print("❗扫描完成，但未能得到有效高度（可能所有文件为空）。")
        return

    print("\n===== Scan Result =====")
    print(f"Scanned files: {num_files} (bad: {num_bad})")
    print(f"Max height (z): {global_max:.6f} m")
    print(f"From file     : {max_file}")

if __name__ == "__main__":
    main()

# python max_height_in_folder.py --folder /path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/raw_lidar/val