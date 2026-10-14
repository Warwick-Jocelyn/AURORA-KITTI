#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D crop for 16-bit sparse depth / projected LiDAR.

KITTI `val_selection_cropped` convention (375x1242 -> 352x1216):
    top  = h - out_h     # keep the BOTTOM of the image
    left = (w - out_w)//2 # center horizontally

This is a PURE SLICE: no interpolation, no resampling. Every retained pixel
keeps its exact 16-bit value and its geometric position, so the cropped sparse
depth stays pixel-aligned with the identically-cropped RGB / groundtruth.

Do NOT use cv2.resize for sparse depth: INTER_AREA/INTER_LINEAR average across
the many zero (empty) pixels and destroy the sparse measurements, and even
INTER_NEAREST moves points to wrong pixels and breaks calibration alignment.
"""
import os
import cv2


def crop_top_lr_16bit(depth_u16, out_h=352, out_w=1216):
    """Pure-slice crop (keep bottom, center horizontally). Values untouched."""
    h, w = depth_u16.shape[:2]
    top, left = h - out_h, (w - out_w) // 2
    assert top >= 0 and left >= 0, f"image {h}x{w} is smaller than target {out_h}x{out_w}"
    return depth_u16[top:top + out_h, left:left + out_w]


def crop_folder(input_folder, output_folder, out_h=352, out_w=1216):
    """Crop every 16-bit single-channel PNG in input_folder into output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    total = done = skipped = 0
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(".png"):
            continue
        total += 1
        depth = cv2.imread(os.path.join(input_folder, fname), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != "uint16" or depth.ndim != 2:
            print(f"[skip] not single-channel uint16: {fname}")
            skipped += 1
            continue
        cropped = crop_top_lr_16bit(depth, out_h, out_w)
        cv2.imwrite(os.path.join(output_folder, fname), cropped)
        done += 1
    print(f"[crop_folder] total={total} cropped={done} skipped={skipped} -> {output_folder}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_h", type=int, default=352)
    ap.add_argument("--out_w", type=int, default=1216)
    a = ap.parse_args()
    crop_folder(a.in_dir, a.out_dir, a.out_h, a.out_w)
