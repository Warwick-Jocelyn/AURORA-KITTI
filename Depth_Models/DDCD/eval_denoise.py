#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-agnostic DCD denoising-metric evaluation (ERR/SR, DCE, NRR/CPP), broken
down by weather x severity x time_of_day.

It scores a directory of PREDICTED dense depth maps against the paired
clean/weather sparse LiDAR + dense GT that AURORA-KITTI provides, so the SAME
script evaluates DDCD, the modality ablations, and every baseline (just point
--pred_dir at each method's saved predictions).

Inputs per frame (TEST/VAL set):
  GT  : groundtruth_depth/<drive>_groundtruth_depth_<frame>_<cam>.png   (/256)
  S_w : velodyne_raw/<drive>_velodyne_raw_<frame>_<cam>.png             (/255, weather)
  S_c : velodyne_raw_clean/<drive>_velodyne_raw_<frame>_<cam>.png       (/255, clean)
  P   : <pred_dir>/<gt_basename>                                        (/--pred_scale)

Quick self-tests (no model needed, validate plumbing):
  python eval_denoise.py --self_test perfect   # expect ERR=1, DCE=0
  python eval_denoise.py --self_test copy       # expect ERR=0 (P=S_w)

Metric definitions: DCD_denoising_metrics_design.md
"""
import os
import sys
import csv
import argparse
import numpy as np
import cv2

_THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _THIS)
from criteria import DenoiseMetrics  # noqa: E402

KBASE = os.path.join(_THIS, "datas", "kitti")
SETS = {
    "TEST": ("val_selection_cropped_TEST", "test_small.csv"),
    "VAL":  ("val_selection_cropped_VAL",  "val_small.csv"),
}
WEATHER_ORDER = ["Clear", "Fog", "Rain", "Snow"]
SEV_ORDER = ["None", "light", "moderate", "heavy"]
TOD_ORDER = ["Day", "Night"]


def load_depth(path, scale):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="TEST", choices=list(SETS))
    ap.add_argument("--pred_dir", default="", help="dir of predicted dense depth PNGs (named like the GT files)")
    ap.add_argument("--pred_scale", type=float, default=256.0, help="pred PNG encoding (meters*scale)")
    ap.add_argument("--sparse_scale", type=float, default=255.0, help="velodyne_raw/clean encoding (our sets = *255)")
    ap.add_argument("--gt_scale", type=float, default=256.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--self_test", default="none", choices=["none", "perfect", "copy"],
                    help="perfect: P=GT ; copy: P=S_w (no pred_dir needed)")
    ap.add_argument("--out_prefix", default="denoise_breakdown")
    args = ap.parse_args()

    sub, csvn = SETS[args.set]
    root = os.path.join(KBASE, sub)
    clean_dir = os.path.join(root, "velodyne_raw_clean")
    weather_dir = os.path.join(root, "velodyne_raw")
    gt_dir = os.path.join(root, "groundtruth_depth")

    # weather meta keyed by (drive, cam, frame)
    meta = {}
    for r in csv.DictReader(open(os.path.join(root, csvn))):
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        drive, rest = r["name"].split("_image_"); camid, frame = rest.split("_")
        meta[(drive, f"image_{camid}", frame)] = r

    dm = DenoiseMetrics(tau=args.tau)
    n_done = n_skip = 0
    for fn in sorted(os.listdir(clean_dir)):
        # fn = <drive>_velodyne_raw_<frame>_image_<cam>.png
        stem = fn[:-4]
        try:
            drive, rest = stem.split("_velodyne_raw_"); frame, cam = rest.split("_image_")
            cam = "image_" + cam
        except ValueError:
            n_skip += 1; continue
        key = (drive, cam, frame)
        m = meta.get(key)
        if m is None:
            n_skip += 1; continue

        S_c = load_depth(os.path.join(clean_dir, fn), args.sparse_scale)
        S_w = load_depth(os.path.join(weather_dir, fn), args.sparse_scale)
        G = load_depth(os.path.join(gt_dir, fn.replace("_velodyne_raw_", "_groundtruth_depth_")), args.gt_scale)
        if S_c is None or S_w is None or G is None:
            n_skip += 1; continue

        if args.self_test == "perfect":
            P = G.copy()
        elif args.self_test == "copy":
            P = S_w.copy()
        else:
            gt_name = fn.replace("_velodyne_raw_", "_groundtruth_depth_")
            P = load_depth(os.path.join(args.pred_dir, gt_name), args.pred_scale)
            if P is None:  # try same-name-as-velodyne fallback
                P = load_depth(os.path.join(args.pred_dir, fn), args.pred_scale)
            if P is None:
                n_skip += 1; continue
        if P.shape != G.shape:
            P = cv2.resize(P, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)

        w = m.get("weather", "?"); s = m.get("severity", "?") or "None"; t = m.get("Time_of_Day", "?")
        dm.add(P, G, S_w, S_c, groups=["ALL", w, (w, s), (w, t)])
        n_done += 1

    print(f"[{args.set}] scored={n_done} skipped={n_skip}  tau={args.tau} self_test={args.self_test}")

    # ---- assemble rows: overall, per weather, per weather+severity, per weather+tod ----
    def row(label, g):
        if g not in dm.acc:
            return None
        r = dm.reduce(g)
        return [label, r['ERR'], r['SR'], r['DCE'] * 1000, r['DCE_mae'] * 1000, r['NRR'], r['CPP'] * 1000,
                r['n_cor'], r['n_drop'], r['n_keep']]
    rows = [row("ALL", "ALL")]
    rows += [row(w, w) for w in WEATHER_ORDER]
    rows += [row(f"{w}/{s}", (w, s)) for w in WEATHER_ORDER for s in SEV_ORDER]
    rows += [row(f"{w}/{t}", (w, t)) for w in WEATHER_ORDER for t in TOD_ORDER]
    rows = [r for r in rows if r is not None]

    hdr = ["group", "ERR", "SR", "DCE_mm", "DCE_mae_mm", "NRR", "CPP_mm", "n_cor", "n_drop", "n_keep"]
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(hdr)
        for r in rows:
            wtr.writerow([r[0]] + [f"{x:.4f}" if isinstance(x, float) else x for x in r[1:]])

    def cell(x):
        return f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else ("—" if isinstance(x, float) else str(x))
    md = [f"# {args.set} — DCD denoising metrics (tau={args.tau})\n",
          "ERR↑ (denoising of corrupted returns), SR↑ (suppression rate), DCE↓ (dropout completion RMSE/MAE, mm), "
          "NRR→1 (noise/clean error ratio), CPP (clean-point error, mm). Pixel-pooled. "
          "See `DCD_denoising_metrics_design.md`.\n",
          "| Group | ERR↑ | SR↑ | DCE↓(mm) | DCE-MAE(mm) | NRR→1 | CPP(mm) | n_cor | n_drop | n_keep |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join([r[0]] + [cell(x) for x in r[1:]]) + " |")
    open(f"{args.out_prefix}.md", "w").write("\n".join(md) + "\n")
    print(f"wrote {args.out_prefix}.csv / .md")


if __name__ == "__main__":
    main()
