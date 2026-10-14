#!/usr/bin/env python
"""
Snow-severity rescue attempt (targeted follow-up to Deliverable 1 / monotonicity.json).

The global-pooled statistics in compute_deliverable1.py (dropout_rate, preserved_rate,
range_rmse_pert_m, near_range_clutter_ratio, ...) barely separate across snow
light/moderate/heavy, unlike fog/rain which are cleanly monotonic. This script tests
whether snow severity IS encoded in the paired AURORA sparse LiDAR (S_c=velodyne_raw_clean,
S_w=velodyne_raw) under a handful of ALTERNATIVE statistics that are finer-grained than
the global pooled rates:

  1. Point-density-vs-distance retention curve: per range-bin ratio of valid weather
     points to valid clean points (bins: 0-10,10-20,...,50-60,>60 m).
  2. Dropout rate AS A FUNCTION OF RANGE (paired: bin by the clean/true range S_c,
     check whether that pixel survives in S_w). This is the physically principled
     twin of (1) -- (1) bins weather points by their OWN (possibly perturbed) range,
     independently from clean; (2) is a per-pixel paired dropout curve.
  3. Spurious-point spatial structure: per-severity mean range of spurious (M_spur)
     points, spurious count-per-frame, and a fine-grained (2.5 m bin) histogram of
     spurious-point ranges, to check for a distinct near/mid-range clutter population.
  4. Intensity: AURORA sparse LiDAR pipeline only ever exposes DEPTH (uint16 PNG /256,
     confirmed exact match against velodyne_raw_npy_in_meter_AURORA in Deliverable 1;
     that side-car is also plain depth in meters, no separate channel). No raw .bin
     point clouds and no intensity channel anywhere under DMD3C/BP-Net/DMD3C/datas/kitti.
     SKIPPED -- reported as unavailable, not silently omitted.
  5. Perturbed range-error DISTRIBUTION (not just RMSE): p50/p75/p90/p95/p99/max of
     |S_w - S_c| over M_pert pixels, per severity.

Applied to snow (light/moderate/heavy) AND fog (light/moderate/heavy) as a positive
control -- if a statistic can't even order fog, it isn't a valid severity probe.

CPU-only. Read-only on all dataset paths. Reuses helpers from compute_deliverable1.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_deliverable1 import (  # noqa: E402
    DATA_ROOT, VAL_DIR, TEST_DIR, T_VALID, TAU,
    load_depth_png, load_split_rows,
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

RANGE_BIN_EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 1e9])
RANGE_BIN_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", ">60"]
FINE_HIST_EDGES = np.arange(0, 105, 2.5)  # for spurious-range histogram, matches deliverable1

WEATHERS_OF_INTEREST = ("snow", "fog")
SEVERITIES = ("light", "moderate", "heavy")


class RescueAccum:
    """Pixel-pooled accumulator for one (weather, severity) cell, alternative stats."""

    def __init__(self):
        nb = len(RANGE_BIN_EDGES) - 1
        self.n_frames = 0
        self.n_Mc = 0
        self.n_Mw = 0
        self.n_drop = 0
        self.n_spur = 0
        self.n_pert = 0
        self.n_both = 0

        # (1) + (2): range-binned counts
        self.clean_count_per_bin = np.zeros(nb, dtype=np.int64)     # Mc, binned by S_c (true range)
        self.drop_count_per_bin = np.zeros(nb, dtype=np.int64)      # drop, binned by S_c (true range)
        self.weather_count_per_bin = np.zeros(nb, dtype=np.int64)   # Mw, binned by S_w (own range) -- item(1) literal

        # (3) spurious spatial structure
        self.spur_range_sum = 0.0
        self.spur_per_frame = []
        self.spur_range_hist = np.zeros(len(FINE_HIST_EDGES) - 1, dtype=np.int64)

        # (5) perturbed error distribution -- store raw diffs (small population, cheap)
        self.pert_diffs = []

    def update(self, sc, sw):
        Mc = sc > T_VALID
        Mw = sw > T_VALID
        drop = Mc & (~Mw)
        spur = Mw & (~Mc)
        both = Mc & Mw
        diff = np.abs(sc - sw)
        tol = TAU * np.maximum(sc, 1e-6)
        pert = both & (diff > tol)

        self.n_frames += 1
        self.n_Mc += int(Mc.sum())
        self.n_Mw += int(Mw.sum())
        self.n_drop += int(drop.sum())
        self.n_spur += int(spur.sum())
        self.n_both += int(both.sum())
        self.n_pert += int(pert.sum())

        nb = len(RANGE_BIN_EDGES) - 1
        if Mc.any():
            c_bins = np.clip(np.digitize(sc[Mc], RANGE_BIN_EDGES) - 1, 0, nb - 1)
            self.clean_count_per_bin += np.bincount(c_bins, minlength=nb)
        if drop.any():
            d_bins = np.clip(np.digitize(sc[drop], RANGE_BIN_EDGES) - 1, 0, nb - 1)
            self.drop_count_per_bin += np.bincount(d_bins, minlength=nb)
        if Mw.any():
            w_bins = np.clip(np.digitize(sw[Mw], RANGE_BIN_EDGES) - 1, 0, nb - 1)
            self.weather_count_per_bin += np.bincount(w_bins, minlength=nb)

        n_spur_frame = int(spur.sum())
        self.spur_per_frame.append(n_spur_frame)
        if n_spur_frame > 0:
            spur_ranges = sw[spur]
            self.spur_range_sum += float(spur_ranges.sum())
            h, _ = np.histogram(spur_ranges, bins=FINE_HIST_EDGES)
            self.spur_range_hist += h.astype(np.int64)

        if pert.any():
            self.pert_diffs.append(diff[pert].copy())

    def summarize(self):
        def safe_div(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            out = np.full_like(a, np.nan)
            nz = b > 0
            out[nz] = a[nz] / b[nz]
            return out

        dropout_per_bin = safe_div(self.drop_count_per_bin, self.clean_count_per_bin)
        retention_per_bin_paired = 1.0 - dropout_per_bin  # (2)
        retention_per_bin_literal = safe_div(self.weather_count_per_bin, self.clean_count_per_bin)  # (1)

        pert_all = np.concatenate(self.pert_diffs) if self.pert_diffs else np.array([])
        pert_q = {}
        if pert_all.size > 0:
            for q in (50, 75, 90, 95, 99):
                pert_q[f"p{q}"] = float(np.percentile(pert_all, q))
            pert_q["mean"] = float(np.mean(pert_all))
            pert_q["max"] = float(np.max(pert_all))
            pert_q["n"] = int(pert_all.size)
        else:
            pert_q = {"p50": None, "p75": None, "p90": None, "p95": None, "p99": None,
                       "mean": None, "max": None, "n": 0}

        spur_per_frame_arr = np.array(self.spur_per_frame, dtype=float)

        return {
            "n_frames": self.n_frames,
            "n_Mc_px": self.n_Mc,
            "n_Mw_px": self.n_Mw,
            "dropout_rate_global": (self.n_drop / self.n_Mc) if self.n_Mc > 0 else None,
            "range_bin_labels": RANGE_BIN_LABELS,
            "range_bin_edges_m": RANGE_BIN_EDGES.tolist(),
            "clean_count_per_bin": self.clean_count_per_bin.tolist(),
            "drop_count_per_bin": self.drop_count_per_bin.tolist(),
            "weather_count_per_bin": self.weather_count_per_bin.tolist(),
            "dropout_rate_per_bin_paired": [None if np.isnan(x) else float(x) for x in dropout_per_bin],
            "retention_per_bin_paired": [None if np.isnan(x) else float(x) for x in retention_per_bin_paired],
            "retention_per_bin_literal_item1": [None if np.isnan(x) else float(x) for x in retention_per_bin_literal],
            "spurious": {
                "n_spur_total": self.n_spur,
                "mean_range_m": (self.spur_range_sum / self.n_spur) if self.n_spur > 0 else None,
                "count_per_frame_mean": float(spur_per_frame_arr.mean()) if spur_per_frame_arr.size else None,
                "count_per_frame_std": float(spur_per_frame_arr.std()) if spur_per_frame_arr.size else None,
                "range_hist_edges_m": FINE_HIST_EDGES.tolist(),
                "range_hist_counts": self.spur_range_hist.tolist(),
            },
            "perturbed_error_quantiles_m": pert_q,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit frames per split for smoke test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "snow_rescue_raw.json"))
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
    all_rows = [r for r in all_rows if r["weather"] in WEATHERS_OF_INTEREST and r["severity"] in SEVERITIES]
    print(f"Rows in scope (weather in {WEATHERS_OF_INTEREST}, severity in {SEVERITIES}): {len(all_rows)}")

    if args.dry_run:
        from collections import defaultdict
        cells = defaultdict(int)
        for r in all_rows:
            cells[(r["weather"], r["severity"])] += 1
        for k, v in sorted(cells.items()):
            print(k, v)
        return

    from collections import defaultdict
    cells = defaultdict(RescueAccum)
    n_missing = 0
    for i, r in enumerate(all_rows):
        if not (os.path.exists(r["sw_path"]) and os.path.exists(r["sc_path"])):
            n_missing += 1
            continue
        sw = load_depth_png(r["sw_path"])
        sc = load_depth_png(r["sc_path"])
        if sw.shape != sc.shape:
            continue
        cells[(r["weather"], r["severity"])].update(sc, sw)
        if (i + 1) % 400 == 0:
            print(f"  processed {i+1}/{len(all_rows)}")
    print(f"missing files: {n_missing}")

    out = {
        "meta": {
            "t_valid": T_VALID,
            "tau": TAU,
            "range_bin_labels": RANGE_BIN_LABELS,
            "range_bin_edges_m": RANGE_BIN_EDGES.tolist(),
            "fine_hist_edges_m": FINE_HIST_EDGES.tolist(),
            "weathers": WEATHERS_OF_INTEREST,
            "severities": SEVERITIES,
            "intensity_available": False,
            "intensity_note": (
                "No intensity channel found under DMD3C/BP-Net/DMD3C/datas/kitti. "
                "velodyne_raw/velodyne_raw_clean are uint16 depth-only PNGs (m*256); "
                "velodyne_raw_npy_in_meter_AURORA side-car confirmed to be the same "
                "depth values in meters (single-channel float32), not intensity. "
                "No raw .bin point clouds present. Statistic (4) skipped."
            ),
        },
        "cells": {f"{w}|{s}": acc.summarize() for (w, s), acc in cells.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
