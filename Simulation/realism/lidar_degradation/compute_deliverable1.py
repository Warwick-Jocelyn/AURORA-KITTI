#!/usr/bin/env python
"""
Deliverable 1 (RC3.3): dataset-level LiDAR degradation characterization for AURORA-KITTI.

Reuses the paper's own DCD noise masks (Denoising_metrics.tex, Eqs. for M_c/M_w/M_drop/
M_spur/M_pert/M_keep) but applies them to the INPUT sparse LiDAR (S_c = velodyne_raw_clean,
S_w = velodyne_raw), pooled over VAL+TEST, per weather x severity cell.

Units: velodyne_raw* PNGs are uint16 = depth_meters * 256 (standard KITTI-DC convention).
Confirmed by exact match (0.0 max abs diff) against velodyne_raw_npy_in_meter_AURORA on
5 sampled TEST frames before trusting this scale (see recon notes in findings.md).

CPU-only. Read-only on all dataset paths.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
VAL_DIR = os.path.join(DATA_ROOT, "val_selection_cropped_VAL")
TEST_DIR = os.path.join(DATA_ROOT, "val_selection_cropped_TEST")

T_VALID = 1e-3  # m
TAU = 0.05      # relative perturbation tolerance
NEAR_RANGE_M = 8.0  # near-range clutter threshold

NAME_RE = re.compile(r"(.+)_image_(\d+)_(\d+)$")


def name_to_velofile(name):
    """VAL/TEST csv 'name' -> velodyne_raw filename.
    name:     2011_09_26_drive_0002_sync_image_02_0000000006
    filename: 2011_09_26_drive_0002_sync_velodyne_raw_0000000006_image_02.png
    """
    m = NAME_RE.match(name.strip())
    if m is None:
        raise ValueError(f"unparseable name: {name!r}")
    seq, cam, frame = m.group(1), m.group(2), m.group(3)
    return f"{seq}_velodyne_raw_{frame}_image_{cam}.png"


def load_depth_png(path):
    """uint16 PNG -> float32 depth in meters (KITTI-DC convention: /256)."""
    arr = np.array(Image.open(path), dtype=np.float64)
    return (arr / 256.0).astype(np.float64)


def norm_weather(w):
    w = str(w).strip().lower()
    return w


def norm_severity(s):
    s = str(s).strip().lower()
    if s in ("medium",):
        s = "moderate"
    if s in ("none", "nan", ""):
        s = "clear"
    return s


def load_split_rows(csv_path, velo_dir, clean_dir):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        name = str(r["name"]).strip()
        weather = norm_weather(r["weather"])
        severity = norm_severity(r["severity"])
        fn = name_to_velofile(name)
        rows.append({
            "name": name,
            "weather": weather,
            "severity": severity,
            "sw_path": os.path.join(velo_dir, fn),
            "sc_path": os.path.join(clean_dir, fn),
        })
    return rows


class Accum:
    """Pixel-pooled accumulator for one (weather,severity) cell."""

    def __init__(self):
        self.n_frames = 0
        self.n_Mc = 0
        self.n_Mw = 0
        self.n_drop = 0
        self.n_spur = 0
        self.n_pert = 0
        self.n_keep = 0
        self.n_both = 0  # Mc & Mw
        self.sumsq_pert = 0.0
        self.sumsq_pert_keep = 0.0
        self.n_pert_keep = 0
        # observable stats (no clean needed) computed on S_w
        self.p95_per_frame = []
        self.max_per_frame = []
        self.near_count = 0
        self.valid_w_count = 0
        self.range_hist_counts = None  # filled lazily
        self.n_pixels_total = 0

    def update(self, sc, sw, hist_edges):
        Mc = sc > T_VALID
        Mw = sw > T_VALID
        drop = Mc & (~Mw)
        spur = Mw & (~Mc)
        both = Mc & Mw
        diff = np.abs(sc - sw)
        tol = TAU * np.maximum(sc, 1e-6)
        pert = both & (diff > tol)
        keep = both & (diff <= tol)

        self.n_frames += 1
        self.n_Mc += int(Mc.sum())
        self.n_Mw += int(Mw.sum())
        self.n_drop += int(drop.sum())
        self.n_spur += int(spur.sum())
        self.n_pert += int(pert.sum())
        self.n_keep += int(keep.sum())
        self.n_both += int(both.sum())
        if pert.any():
            self.sumsq_pert += float(np.sum((sw[pert] - sc[pert]) ** 2))
        pk = pert | keep
        if pk.any():
            self.sumsq_pert_keep += float(np.sum((sw[pk] - sc[pk]) ** 2))
            self.n_pert_keep += int(pk.sum())

        # observable stats on S_w only
        valid_w = sw[Mw]
        self.valid_w_count += valid_w.size
        self.n_pixels_total += sw.size
        if valid_w.size > 0:
            self.p95_per_frame.append(float(np.percentile(valid_w, 95)))
            self.max_per_frame.append(float(valid_w.max()))
            self.near_count += int((valid_w < NEAR_RANGE_M).sum())
            h, _ = np.histogram(valid_w, bins=hist_edges)
            if self.range_hist_counts is None:
                self.range_hist_counts = h.astype(np.int64)
            else:
                self.range_hist_counts += h.astype(np.int64)

    def summarize(self):
        def safe_div(a, b):
            return float(a) / float(b) if b > 0 else float("nan")

        range_rmse_pert = (
            float(np.sqrt(self.sumsq_pert / self.n_pert)) if self.n_pert > 0 else float("nan")
        )
        range_rmse_pertkeep = (
            float(np.sqrt(self.sumsq_pert_keep / self.n_pert_keep))
            if self.n_pert_keep > 0
            else float("nan")
        )
        return {
            "n_frames": self.n_frames,
            "n_Mc_px": self.n_Mc,
            "n_Mw_px": self.n_Mw,
            "n_both_px": self.n_both,
            "dropout_rate": safe_div(self.n_drop, self.n_Mc),
            "spurious_rate": safe_div(self.n_spur, self.n_Mw),
            "perturbed_rate": safe_div(self.n_pert, self.n_both),
            "preserved_rate": safe_div(self.n_keep, self.n_Mc),
            "range_rmse_pert_m": range_rmse_pert,
            "range_rmse_pert_keep_m": range_rmse_pertkeep,
            "eff_range_p95_m": (
                float(np.mean(self.p95_per_frame)) if self.p95_per_frame else float("nan")
            ),
            "eff_range_mean_of_frame_max_m": (
                float(np.mean(self.max_per_frame)) if self.max_per_frame else float("nan")
            ),
            "near_range_clutter_ratio": safe_div(self.near_count, self.valid_w_count),
            "valid_w_pixel_count": self.valid_w_count,
            "range_hist_counts": (
                self.range_hist_counts.tolist() if self.range_hist_counts is not None else None
            ),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit frames per split for smoke test")
    ap.add_argument("--dry-run", action="store_true", help="print plan, do not read images")
    ap.add_argument(
        "--out",
        default="/path/to/aurora-root/PhD_Project_2025/Realism_updated/lidar_degradation/deliverable1_raw.json",
    )
    args = ap.parse_args()

    val_rows = load_split_rows(
        os.path.join(VAL_DIR, "val_small.csv"),
        os.path.join(VAL_DIR, "velodyne_raw"),
        os.path.join(VAL_DIR, "velodyne_raw_clean"),
    )
    test_rows = load_split_rows(
        os.path.join(TEST_DIR, "test_small.csv"),
        os.path.join(TEST_DIR, "velodyne_raw"),
        os.path.join(TEST_DIR, "velodyne_raw_clean"),
    )
    if args.limit:
        val_rows = val_rows[: args.limit]
        test_rows = test_rows[: args.limit]

    all_rows = [dict(r, split="VAL") for r in val_rows] + [dict(r, split="TEST") for r in test_rows]
    print(f"Loaded {len(val_rows)} VAL rows + {len(test_rows)} TEST rows = {len(all_rows)} total")

    if args.dry_run:
        cells = defaultdict(int)
        for r in all_rows:
            cells[(r["weather"], r["severity"])] += 1
        for k, v in sorted(cells.items()):
            print(k, v)
        return

    hist_edges = np.arange(0, 105, 2.5)  # 0..102.5m in 2.5m bins

    cells = defaultdict(Accum)
    n_missing = 0
    skipped = []
    for i, r in enumerate(all_rows):
        if not (os.path.exists(r["sw_path"]) and os.path.exists(r["sc_path"])):
            n_missing += 1
            skipped.append({**r, "reason": "missing_file"})
            continue
        sw = load_depth_png(r["sw_path"])
        sc = load_depth_png(r["sc_path"])
        if sw.shape != sc.shape:
            skipped.append({**r, "reason": f"shape_mismatch {sw.shape} vs {sc.shape}"})
            continue
        cells[(r["weather"], r["severity"])].update(sc, sw, hist_edges)
        if (i + 1) % 400 == 0:
            print(f"  processed {i+1}/{len(all_rows)}")

    print(f"missing files: {n_missing}")

    out = {
        "meta": {
            "t_valid": T_VALID,
            "tau": TAU,
            "near_range_m": NEAR_RANGE_M,
            "hist_edges_m": hist_edges.tolist(),
            "n_total_rows": len(all_rows),
            "n_missing": n_missing,
            "depth_scale": "uint16_png / 256.0 -> meters (verified against velodyne_raw_npy_in_meter_AURORA, 0.0 max abs diff, 5 TEST frames)",
        },
        "cells": {f"{w}|{s}": acc.summarize() for (w, s), acc in cells.items()},
        "skipped": skipped,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")

    if skipped:
        import csv

        skip_csv = os.path.join(os.path.dirname(args.out), "skipped_deliverable1.csv")
        with open(skip_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "name", "weather", "severity", "reason"])
            w.writeheader()
            for s in skipped:
                w.writerow({k: s.get(k) for k in ["split", "name", "weather", "severity", "reason"]})
        print(f"Wrote {skip_csv} ({len(skipped)} rows)")


if __name__ == "__main__":
    main()
