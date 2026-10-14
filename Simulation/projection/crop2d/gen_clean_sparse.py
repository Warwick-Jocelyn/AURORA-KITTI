#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the CLEAN sparse-LiDAR reference (velodyne_raw_clean/) that is
pixel-aligned with the existing weather-corrupted velodyne_raw/.

Method: re-project the clean 3D point clouds (Clear_Lidar) with the SAME
projection pipeline that produced the weather set, then apply the same
352x1216 crop. On `Clear` weather frames this reproduces velodyne_raw/
byte-for-byte (self-checked); on fog/rain/snow it differs ONLY by the
weather-induced dropout -- exactly what denoising metrics need.

Encoding: depth is stored as meters*255 ON PURPOSE, to match the existing
weather set (which was generated at *255). This keeps clean and corrupted on
the same scale so the *255-vs-*256 question cancels in any comparison.
(The library bug-fix to *256 is for FUTURE regenerations only.)
"""
import os
import sys
import csv
import numpy as np
import cv2

PROJ_DIR = "/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Codes_Lidar_Generation/Projection"
sys.path.insert(0, PROJ_DIR)
from lidar_projection_utils import load_from_bin, velo_points_filter  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from crop2d import crop_top_lr_16bit  # noqa: E402

CLEAR_ROOT = "/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/Clear_Lidar"
CALIB_ROOT = "/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Codes_Lidar_Generation/calib"
KITTI_BASE = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
SETS = {
    "TEST": ("val_selection_cropped_TEST", "test_small.csv"),
    "VAL":  ("val_selection_cropped_VAL",  "val_small.csv"),
}
IMG_H, IMG_W = 375, 1242
SCALE = 255.0  # match existing weather set on purpose (see module docstring)


def parse_calib(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                d[k] = np.fromstring(v, sep=" ")
            except Exception:
                pass
    return d


_calib_cache = {}


def get_calib(date_str, cam):
    key = (date_str, cam)
    if key in _calib_cache:
        return _calib_cache[key]
    cdir = os.path.join(CALIB_ROOT, f"{date_str}_calib")
    vc = parse_calib(os.path.join(cdir, "calib_velo_to_cam.txt"))
    cc = parse_calib(os.path.join(cdir, "calib_cam_to_cam.txt"))
    RT = np.hstack((vc["R"].reshape(3, 3), vc["T"].reshape(3, 1)))
    R_rect = cc["R_rect_00"].reshape(3, 3)
    P = cc[f"P_rect_{cam}"].reshape(3, 4)
    _calib_cache[key] = (RT, R_rect, P)
    return _calib_cache[key]


def project_depth(velo, RT, R_rect, P):
    """Replicate create_depth_map (nearest wins) vectorized, at SCALE. uint16."""
    xyz_hom, _ = velo_points_filter(velo[:, :3], v_fov=(-24.9, 3.0), h_fov=(-180, 180))
    xyz_rect = R_rect @ (RT @ xyz_hom)
    depth_c = xyz_rect[2, :].copy()
    uvh = P @ np.vstack((xyz_rect, np.ones((1, xyz_rect.shape[1]))))
    uv = uvh[:2, :] / uvh[2, :]
    x = uv[0, :].astype(np.int64)
    y = uv[1, :].astype(np.int64)
    m = (x >= 0) & (x < IMG_W) & (y >= 0) & (y < IMG_H) & (depth_c > 0)
    x, y, dv = x[m], y[m], depth_c[m]
    dm = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    # nearest wins: assign in descending-depth order so the smallest depth lands last
    order = np.argsort(-dv)
    flat = (y[order] * IMG_W + x[order])
    vals = (dv[order] * SCALE).astype(np.uint16)
    dm.reshape(-1)[flat] = vals
    return dm


def run(which):
    sub, csvname = SETS[which]
    src_csv = os.path.join(KITTI_BASE, sub, csvname)
    weather_dir = os.path.join(KITTI_BASE, sub, "velodyne_raw")
    out_dir = os.path.join(KITTI_BASE, sub, "velodyne_raw_clean")
    os.makedirs(out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(src_csv)))
    n_ok = n_miss_bin = 0
    clear_checked = clear_identical = clear_bad = 0
    bad_examples = []

    for r in rows:
        name = r["name"]                      # 2011_09_26_drive_0011_sync_image_02_0000000005
        weather = r["weather"].strip()
        drive, rest = name.split("_image_")
        cam, frame = rest.split("_")          # '02', '0000000005'
        date_str = "_".join(drive.split("_")[:3])
        bin_path = os.path.join(CLEAR_ROOT, drive, "velodyne_points", "data", f"{frame}.bin")
        if not os.path.isfile(bin_path):
            n_miss_bin += 1
            if len(bad_examples) < 10:
                bad_examples.append(f"MISSING bin: {bin_path}")
            continue

        RT, R_rect, P = get_calib(date_str, cam)
        dm_full = project_depth(load_from_bin(bin_path), RT, R_rect, P)
        dm = crop_top_lr_16bit(dm_full)        # 352x1216

        out_name = f"{drive}_velodyne_raw_{frame}_image_{cam}.png"
        cv2.imwrite(os.path.join(out_dir, out_name), dm)
        n_ok += 1

        # self-check: clear frames must equal the existing weather set byte-for-byte
        if weather == "Clear":
            wp = os.path.join(weather_dir, out_name)
            if os.path.isfile(wp):
                W = cv2.imread(wp, cv2.IMREAD_UNCHANGED)
                clear_checked += 1
                if np.array_equal(dm, W):
                    clear_identical += 1
                else:
                    clear_bad += 1
                    if len(bad_examples) < 10:
                        bad_examples.append(f"CLEAR MISMATCH: {out_name} "
                                            f"(clean>{int((dm>0).sum())} weather>{int((W>0).sum())})")

    print(f"\n===== {which} =====")
    print(f"written clean frames : {n_ok}")
    print(f"missing Clear_Lidar  : {n_miss_bin}")
    print(f"clear self-check     : checked={clear_checked} identical={clear_identical} mismatch={clear_bad}")
    for e in bad_examples:
        print("   ", e)
    return clear_bad == 0 and n_miss_bin == 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "TEST"
    run(which)
