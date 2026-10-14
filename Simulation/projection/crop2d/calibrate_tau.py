#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate the range-perturbation threshold tau for the DCD denoising metrics.

Idea: two INDEPENDENT clean LiDAR projections of the same scenes should agree
up to projection/occlusion effects. Their disagreement is the noise floor below
which a "perturbation" is meaningless. We compare:
  A = our clean projection   (velodyne_raw_clean, x255)   -- gen_clean_sparse.py
  B = provided clean sparse   (official KITTI data_depth_velodyne/velodyne_raw, x256)
both reduced to 352x1216 with the SAME crop, then report the |A-B| distribution
(absolute and relative). tau is taken at ~p99 -> 5% of depth (~0.5 m).
"""
import os, csv, numpy as np, cv2

KBASE = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
OFF   = "/path/to/aurora-root/PhD_Project_2025/kitti_depth/data_depth_velodyne"
SET   = ("val_selection_cropped_TEST", "test_small.csv")
N_FRAMES = 300
OUT_H, OUT_W = 352, 1216


def off_crop(drive, cam, frame):
    for sp in ("train", "val"):
        p = f"{OFF}/{sp}/{drive}/proj_depth/velodyne_raw/image_{cam}/{frame}.png"
        if os.path.exists(p):
            O = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            h, w = O.shape
            return O[h - OUT_H:h, (w - OUT_W) // 2:(w - OUT_W) // 2 + OUT_W]
    return None


def main():
    sub, csvn = SET
    rows = list(csv.DictReader(open(f"{KBASE}/{sub}/{csvn}")))
    absd, reld, npix, nfr = [], [], 0, 0
    for r in rows[:N_FRAMES]:
        drive, rest = r["name"].split("_image_"); cam, frame = rest.split("_")
        ours = cv2.imread(f"{KBASE}/{sub}/velodyne_raw_clean/"
                          f"{drive}_velodyne_raw_{frame}_image_{cam}.png", cv2.IMREAD_UNCHANGED)
        off = off_crop(drive, cam, frame)
        if ours is None or off is None:
            continue
        A = ours.astype(np.float64) / 255.0    # our clean, x255
        B = off.astype(np.float64) / 256.0     # provided clean, x256
        m = (A > 0) & (B > 0)
        if not m.any():
            continue
        d = np.abs(A[m] - B[m])
        absd.append(d); reld.append(d / np.maximum(B[m], 1e-6)); npix += int(m.sum()); nfr += 1
    absd = np.concatenate(absd); reld = np.concatenate(reld) * 100
    print(f"frames={nfr}  matched clean-vs-clean pixels={npix}")
    print("pct |  abs(m) | rel(%)")
    for p in (50, 75, 90, 95, 97.5, 99, 99.5):
        print(f"p{p:<5} | {np.percentile(absd, p):7.3f} | {np.percentile(reld, p):6.2f}")
    print(f"frac>0.5m={ (absd>0.5).mean()*100:.2f}%  frac>5%={ (reld>5).mean()*100:.2f}%  "
          f"frac>3%={ (reld>3).mean()*100:.2f}%")
    print("=> tau = 5% of depth (~0.5 m) sits at ~p99 of inherent clean-vs-clean disagreement.")


if __name__ == "__main__":
    main()
