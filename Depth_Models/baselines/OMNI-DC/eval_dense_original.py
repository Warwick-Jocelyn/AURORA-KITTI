#!/usr/bin/env python3
"""
Evaluate OMNI-DC predictions on original DENSE dataset.

Pred:  output_DENSE_Original/{name}.png   (uint16, pixel = depth_m * 1000)
GT:    GT_renamed/{name}.png              (uint16, pixel = depth_m * 256)

Reports: RMSE, MAE, iRMSE, iMAE (+ REL, delta accuracies)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# ---- paths ----------------------------------------------------------------
PRED_FOLDER = "/path/to/aurora-root/Code/OMNI-DC/output_DENSE_Original"
GT_FOLDER   = "/path/to/aurora-root/Dataset/PixelAccurateDepthBenchmark/datas/GT_renamed"

# ---- scale factors (pixel -> metres) --------------------------------------
PRED_SCALE = 1000.0   # OMNI-DC 16-bit output: pixel = depth_m * 1000
GT_SCALE   = 256.0    # DENSE GT: pixel = depth_m * 256

# ---- valid depth range ----------------------------------------------------
MIN_DEPTH  = 1e-5
MAX_DEPTH  = 80.0

# ---- inverse depth scale --------------------------------------------------
INV_SCALE  = 1000.0   # 1/km  (set to 1.0 for 1/m)

# --------------------------------------------------------------------------

def compute_metrics(pred_path, gt_path):
    pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt_img   = cv2.imread(gt_path,   cv2.IMREAD_UNCHANGED)

    if pred_img is None or gt_img is None:
        return None

    # Align size if needed (nearest-neighbour to preserve depth values)
    if pred_img.shape != gt_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    depth_pred = pred_img.astype(np.float32) / PRED_SCALE
    depth_gt   = gt_img.astype(np.float32)   / GT_SCALE

    mask = (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)
    if not np.any(mask):
        return None

    valid_pred = depth_pred[mask]
    valid_gt   = depth_gt[mask]

    eps      = 1e-8
    diff     = valid_pred - valid_gt
    diff_abs = np.abs(diff)

    rmse  = np.sqrt(np.mean(diff ** 2))
    mae   = diff_abs.mean()
    rel   = (diff_abs / (valid_gt + eps)).mean()

    inv_pred  = INV_SCALE / (valid_pred + eps)
    inv_gt    = INV_SCALE / (valid_gt   + eps)
    diff_inv  = inv_pred - inv_gt
    irmse = np.sqrt(np.mean(diff_inv ** 2))
    imae  = np.mean(np.abs(diff_inv))

    ratio = np.maximum(valid_gt / (valid_pred + eps),
                       valid_pred / (valid_gt + eps))

    return dict(
        RMSE=rmse, MAE=mae, REL=rel,
        iRMSE=irmse, iMAE=imae,
        D1=np.mean(ratio < 1.25),
        D2=np.mean(ratio < 1.25**2),
        D3=np.mean(ratio < 1.25**3),
    )


def main():
    pred_files = sorted(f for f in os.listdir(PRED_FOLDER) if f.endswith('.png'))
    gt_files   = sorted(f for f in os.listdir(GT_FOLDER)   if f.endswith('.png'))

    # Match by filename
    pred_set = set(pred_files)
    gt_set   = set(gt_files)
    common   = sorted(pred_set & gt_set)

    if not common:
        print("No matching filenames found between pred and GT folders!")
        print(f"  Pred sample: {pred_files[:3]}")
        print(f"  GT sample:   {gt_files[:3]}")
        return

    print(f"Evaluating {len(common)} matched pairs...")
    print(f"  Pred: {PRED_FOLDER}")
    print(f"  GT:   {GT_FOLDER}\n")

    all_metrics = {}
    valid_count = 0

    for fname in tqdm(common):
        res = compute_metrics(
            os.path.join(PRED_FOLDER, fname),
            os.path.join(GT_FOLDER,   fname),
        )
        if res is None:
            continue
        valid_count += 1
        for k, v in res.items():
            all_metrics.setdefault(k, []).append(v)

    if valid_count == 0:
        print("No valid pairs found.")
        return

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    unit_inv = "1/km" if INV_SCALE == 1000.0 else "1/m"

    print(f"\n{'='*45}")
    print(f"OMNI-DC on Original DENSE  ({valid_count} images)")
    print(f"{'='*45}")
    print(f"RMSE   (m)      : {avg['RMSE']:.4f}")
    print(f"MAE    (m)      : {avg['MAE']:.4f}")
    print(f"REL             : {avg['REL']:.4f}")
    print(f"iRMSE ({unit_inv}) : {avg['iRMSE']:.4f}")
    print(f"iMAE  ({unit_inv}) : {avg['iMAE']:.4f}")
    print(f"-" * 30)
    print(f"δ<1.25          : {avg['D1']*100:.2f}%")
    print(f"δ<1.25²         : {avg['D2']*100:.2f}%")
    print(f"δ<1.25³         : {avg['D3']*100:.2f}%")
    print(f"{'='*45}")
    print(f"Scale: Pred/{PRED_SCALE:.0f}, GT/{GT_SCALE:.0f}  |  Max depth: {MAX_DEPTH}m")


if __name__ == "__main__":
    main()
