#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP-Basic (fast / multiscale) depth completion on our TEST/DENSE splits.
Parameter-free classical fill. Decodes sparse LiDAR to true metres (TEST velodyne
encoded *255, DENSE *256), fills, saves metres*256 uint16 named like the GT file.
Output dir: results_zeroshot_benchmark/preds/IPBasic-<fill>__<SET>
"""
import os, sys, glob, argparse, numpy as np, cv2
IPB = "/path/to/aurora-root/Code/ip_basic"
sys.path.append(IPB)
from ip_basic import depth_map_utils_original as dmu

KB = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
SETS = {"TEST": ("val_selection_cropped_TEST", 255.0), "DENSE": ("val_selection_cropped_DENSE", 256.0)}


def vel_name(gt_base, set_):
    if set_ == "TEST":
        return gt_base.replace("_groundtruth_depth_", "_velodyne_raw_")
    return gt_base  # DENSE: same basename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--fill", required=True, choices=["fast", "multiscale"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sub, sscale = SETS[args.set]
    root = os.path.join(KB, sub)
    gt_dir = os.path.join(root, "groundtruth_depth")
    vel_dir = os.path.join(root, "velodyne_raw")
    os.makedirs(args.out, exist_ok=True)
    gts = sorted(f for f in os.listdir(gt_dir) if f.endswith(".png"))
    n_ok = n_skip = 0
    for i, gt_base in enumerate(gts):
        vp = os.path.join(vel_dir, vel_name(gt_base, args.set))
        if not os.path.exists(vp):
            n_skip += 1; continue
        try:
            with open(vp, "rb") as fh:
                sparse = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        except OSError:
            sparse = None
        if sparse is None:
            n_skip += 1; continue
        depth_m = sparse.astype(np.float32) / sscale  # true metres, 0 = empty
        if args.fill == "fast":
            out = dmu.fill_in_fast(depth_m.copy(), max_depth=100.0, extrapolate=True)
        else:
            out, _ = dmu.fill_in_multiscale(depth_m.copy(), max_depth=100.0, extrapolate=True)
        u16 = np.clip(np.round(out * 256.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out, gt_base), u16)
        n_ok += 1
        if (i + 1) % 400 == 0:
            print(f"  [{i+1}/{len(gts)}] {gt_base} range[{out.min():.1f},{out.max():.1f}]m", flush=True)
    print(f"[IPBasic-{args.fill}/{args.set}] saved={n_ok} skipped={n_skip} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
