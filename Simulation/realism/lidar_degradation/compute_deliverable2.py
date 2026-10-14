#!/usr/bin/env python
"""
Deliverable 2 (reviewer-2 "tricky alignment"): severity-matched realism check.

Real DENSE has real weather LiDAR but NO paired clean -> only "observable" stats
(computable from S_w alone) transfer to it: effective range, near-range clutter
ratio, range histogram, point-density-vs-distance.

We compute these for:
  - AURORA-all      (VAL+TEST, weather in {fog,rain}, ALL severities pooled)
  - AURORA-light+moderate (same, but EXCLUDING heavy)
  - DENSE-real      (val_selection_cropped_DENSE, weather in {fog,rain}, all severities)

per weather category (fog, rain), so the comparison is apples-to-apples.

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
DENSE_DIR = os.path.join(DATA_ROOT, "val_selection_cropped_DENSE")

T_VALID = 1e-3
NEAR_RANGE_M = 8.0
HIST_EDGES = np.arange(0, 105, 2.5)
DIST_BIN_EDGES = np.arange(0, 105, 5.0)  # for point-density-vs-distance

NAME_RE = re.compile(r"(.+)_image_(\d+)_(\d+)$")


def name_to_velofile_aurora(name):
    m = NAME_RE.match(name.strip())
    seq, cam, frame = m.group(1), m.group(2), m.group(3)
    return f"{seq}_velodyne_raw_{frame}_image_{cam}.png"


def norm_weather(w):
    return str(w).strip().lower()


def norm_severity(s):
    s = str(s).strip().lower()
    if s in ("medium",):
        s = "moderate"
    if s in ("none", "nan", ""):
        s = "clear"
    return s


def load_depth_png(path):
    arr = np.array(Image.open(path), dtype=np.float64)
    return (arr / 256.0).astype(np.float64)


class ObsAccum:
    def __init__(self):
        self.n_frames = 0
        self.p95_per_frame = []
        self.max_per_frame = []
        self.near_count = 0
        self.valid_count = 0
        self.range_hist = None
        self.dist_point_counts = None  # points per distance bin (for density-vs-distance)

    def update(self, sw):
        valid = sw > T_VALID
        vals = sw[valid]
        self.n_frames += 1
        if vals.size == 0:
            return
        self.valid_count += vals.size
        self.p95_per_frame.append(float(np.percentile(vals, 95)))
        self.max_per_frame.append(float(vals.max()))
        self.near_count += int((vals < NEAR_RANGE_M).sum())
        h, _ = np.histogram(vals, bins=HIST_EDGES)
        d, _ = np.histogram(vals, bins=DIST_BIN_EDGES)
        if self.range_hist is None:
            self.range_hist = h.astype(np.int64)
            self.dist_point_counts = d.astype(np.int64)
        else:
            self.range_hist += h.astype(np.int64)
            self.dist_point_counts += d.astype(np.int64)

    def summarize(self):
        def m(lst):
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "n_frames": self.n_frames,
            "valid_pixel_count": self.valid_count,
            "eff_range_p95_m": m(self.p95_per_frame),
            "eff_range_mean_of_frame_max_m": m(self.max_per_frame),
            "near_range_clutter_ratio": (
                self.near_count / self.valid_count if self.valid_count > 0 else float("nan")
            ),
            "range_hist_counts": self.range_hist.tolist() if self.range_hist is not None else None,
            "dist_bin_point_counts": (
                self.dist_point_counts.tolist() if self.dist_point_counts is not None else None
            ),
            "points_per_frame": (
                self.valid_count / self.n_frames if self.n_frames > 0 else float("nan")
            ),
        }


def load_aurora_rows():
    rows = []
    for split, d, csv_name in [("VAL", VAL_DIR, "val_small.csv"), ("TEST", TEST_DIR, "test_small.csv")]:
        df = pd.read_csv(os.path.join(d, csv_name))
        df.columns = [c.strip() for c in df.columns]
        velo_dir = os.path.join(d, "velodyne_raw")
        for _, r in df.iterrows():
            name = str(r["name"]).strip()
            weather = norm_weather(r["weather"])
            severity = norm_severity(r["severity"])
            fn = name_to_velofile_aurora(name)
            rows.append({
                "split": split, "name": name, "weather": weather, "severity": severity,
                "sw_path": os.path.join(velo_dir, fn),
            })
    return rows


def load_dense_rows():
    df = pd.read_csv(os.path.join(DENSE_DIR, "dense_small.csv"))
    df.columns = [c.strip() for c in df.columns]
    velo_dir = os.path.join(DENSE_DIR, "velodyne_raw")
    rows = []
    for _, r in df.iterrows():
        name = str(r["name"]).strip()
        weather = norm_weather(r["weather"])
        severity = norm_severity(r["severity"])
        rows.append({
            "split": "DENSE", "name": name, "weather": weather, "severity": severity,
            "sw_path": os.path.join(velo_dir, name + ".png"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        default="/path/to/aurora-root/PhD_Project_2025/Realism_updated/lidar_degradation/deliverable2_raw.json",
    )
    args = ap.parse_args()

    aurora_rows = load_aurora_rows()
    dense_rows = load_dense_rows()

    if args.limit:
        aurora_rows = aurora_rows[: args.limit]
        dense_rows = dense_rows[: args.limit]

    weathers = ["fog", "rain"]
    # (group_name, source, predicate) -- source in {"aurora","dense"}
    group_defs = [
        ("AURORA-all", "aurora", lambda r: r["weather"] in weathers),
        ("AURORA-light_moderate", "aurora", lambda r: r["weather"] in weathers and r["severity"] in ("light", "moderate")),
        ("DENSE-real", "dense", lambda r: r["weather"] in weathers),
        # clear-weather baseline: checks whether AURORA/DENSE point-density differs
        # for reasons OTHER than weather (sensor / sampling density mismatch)
        ("AURORA-clear", "aurora", lambda r: r["weather"] == "clear"),
        ("DENSE-clear", "dense", lambda r: r["weather"] == "clear"),
        # per-severity breakdown (finer than light+moderate vs all)
        ("AURORA-fog-light", "aurora", lambda r: r["weather"] == "fog" and r["severity"] == "light"),
        ("AURORA-fog-moderate", "aurora", lambda r: r["weather"] == "fog" and r["severity"] == "moderate"),
        ("AURORA-fog-heavy", "aurora", lambda r: r["weather"] == "fog" and r["severity"] == "heavy"),
        ("AURORA-rain-light", "aurora", lambda r: r["weather"] == "rain" and r["severity"] == "light"),
        ("AURORA-rain-moderate", "aurora", lambda r: r["weather"] == "rain" and r["severity"] == "moderate"),
        ("AURORA-rain-heavy", "aurora", lambda r: r["weather"] == "rain" and r["severity"] == "heavy"),
        ("DENSE-fog-light", "dense", lambda r: r["weather"] == "fog" and r["severity"] == "light"),
        ("DENSE-fog-moderate", "dense", lambda r: r["weather"] == "fog" and r["severity"] == "moderate"),
        ("DENSE-fog-heavy", "dense", lambda r: r["weather"] == "fog" and r["severity"] == "heavy"),
        ("DENSE-rain-light", "dense", lambda r: r["weather"] == "rain" and r["severity"] == "light"),
        ("DENSE-rain-heavy", "dense", lambda r: r["weather"] == "rain" and r["severity"] == "heavy"),
    ]

    if args.dry_run:
        for gname, src_name, pred in group_defs:
            src = dense_rows if src_name == "dense" else aurora_rows
            counts = defaultdict(int)
            for r in src:
                if pred(r):
                    counts[r["weather"]] += 1
            print(gname, dict(counts))
        return

    # accumulate: group -> weather -> ObsAccum
    accs = defaultdict(lambda: defaultdict(ObsAccum))
    n_missing = 0
    for gname, src_name, pred in group_defs:
        src = dense_rows if src_name == "dense" else aurora_rows
        for r in src:
            if not pred(r):
                continue
            if not os.path.exists(r["sw_path"]):
                n_missing += 1
                continue
            sw = load_depth_png(r["sw_path"])
            accs[gname][r["weather"]].update(sw)

    print(f"missing files: {n_missing}")

    out = {
        "meta": {
            "t_valid": T_VALID,
            "near_range_m": NEAR_RANGE_M,
            "hist_edges_m": HIST_EDGES.tolist(),
            "dist_bin_edges_m": DIST_BIN_EDGES.tolist(),
        },
        "groups": {
            gname: {w: acc.summarize() for w, acc in wdict.items()}
            for gname, wdict in accs.items()
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
