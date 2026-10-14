#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified, model-agnostic evaluation of ONE method on ONE split.

Computes, in a single pass over the split's groundtruth_depth/ files:
  * standard KITTI metrics  : RMSE, MAE, iRMSE, iMAE  (per-image then averaged)
  * proposed DCD denoising  : ERR, SR, DCE, DCE_mae, NRR, CPP  (pixel-pooled)
Breakdown groups: ALL, per-weather, per-weather/severity, per-weather/time-of-day.

Proposed metrics need paired clean LiDAR (velodyne_raw_clean), which only the
AURORA-KITTI TEST split has. On DENSE (no clean LiDAR) only standard metrics
are produced; proposed fields are omitted.

Predictions are located from each GT basename via --name_map and decoded with
--pred_scale. Mismatched shapes are nearest-resized to GT. Output: a JSON of
all groups -> {metric: value, n: count} at --out.

Usage:
  python eval_one.py --name BPNet --set TEST  --pred_dir <dir> --name_map gt --pred_scale 256 --out <json>
  python eval_one.py --name OMNI  --set DENSE --pred_dir <dir> --name_map gt --pred_scale 1000 --out <json>
"""
import os, sys, csv, json, argparse, numpy as np, cv2

DMD3C = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C"
sys.path.insert(0, DMD3C)
from criteria import DenoiseMetrics  # noqa: E402

KBASE = os.path.join(DMD3C, "datas", "kitti")
SETS = {"TEST": ("val_selection_cropped_TEST", "test_small.csv"),
        "DENSE": ("val_selection_cropped_DENSE", "dense_small.csv")}
MAXD = 80.0
WEATHER_ORDER = ["Clear", "Fog", "Rain", "Snow"]
SEV_ORDER = ["light", "moderate", "heavy"]
TOD_ORDER = ["Day", "Night"]


def load(path, scale):
    # NOTE: cv2.imread (C fopen) triggers ~1s/file on-access AV scan on this host;
    # open().read()+imdecode is ~6x faster (different open path). Always use this.
    try:
        with open(path, "rb") as fh:
            buf = fh.read()
    except OSError:
        return None
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64) / scale


def map_pred(gt_base, name_map):
    """gt_base = groundtruth filename. Return predicted filename for this method."""
    if name_map == "gt":
        return gt_base
    if name_map == "velodyne":
        return gt_base.replace("_groundtruth_depth_", "_velodyne_raw_")
    if name_map == "image":
        return gt_base.replace("_groundtruth_depth_", "_image_")
    if name_map == "unik3d":
        return gt_base.replace("_groundtruth_depth_", "_image_")[:-4] + "_depth_x256.png"
    raise ValueError(name_map)


def std_metrics(P, G):
    m = (G > 1e-3) & (G < MAXD)
    if m.sum() == 0:
        return None
    p = np.clip(P[m], 1e-3, None)
    g = G[m]
    rmse = np.sqrt(np.mean((p - g) ** 2)) * 1000.0
    mae = np.mean(np.abs(p - g)) * 1000.0
    irmse = np.sqrt(np.mean((1.0 / p - 1.0 / g) ** 2)) * 1000.0
    imae = np.mean(np.abs(1.0 / p - 1.0 / g)) * 1000.0
    return rmse, mae, irmse, imae


def groups_of(weather, sev, tod):
    g = ["ALL", weather, (weather, sev), (weather, tod)]
    return g


def gkey(g):
    return g if isinstance(g, str) else "/".join(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--name_map", default="gt", choices=["gt", "velodyne", "image", "unik3d"])
    ap.add_argument("--pred_scale", type=float, default=256.0)
    ap.add_argument("--sparse_scale", type=float, default=255.0)
    ap.add_argument("--gt_scale", type=float, default=256.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub, csvn = SETS[args.set]
    root = os.path.join(KBASE, sub)
    gt_dir = os.path.join(root, "groundtruth_depth")
    clean_dir = os.path.join(root, "velodyne_raw_clean")
    weather_dir = os.path.join(root, "velodyne_raw")
    has_clean = os.path.isdir(clean_dir)
    do_proposed = (args.set == "TEST") and has_clean

    # ---- meta keyed by GT basename ----
    meta = {}
    rows = list(csv.DictReader(open(os.path.join(root, csvn))))
    if args.set == "TEST":
        # csv name = <drive>_image_<cam>_<frame> ; GT = <drive>_groundtruth_depth_<frame>_image_<cam>.png
        for r in rows:
            nm = r["name"].strip()
            drive, rest = nm.split("_image_")
            cam, frame = rest.split("_")
            gt_base = f"{drive}_groundtruth_depth_{frame}_image_{cam}.png"
            meta[gt_base] = r
    else:
        for r in rows:
            meta[r["name"].strip() + ".png"] = r

    # ---- standard accumulators (per-image then average) ----
    std_acc = {}  # gkey -> [n, sum_rmse, sum_mae, sum_irmse, sum_imae]
    dm = DenoiseMetrics(tau=args.tau) if do_proposed else None

    n_done = n_missing_pred = n_no_meta = 0
    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".png"))
    for gt_base in gt_files:
        r = meta.get(gt_base)
        if r is None:
            n_no_meta += 1
            continue
        weather = (r.get("weather") or "?").strip()
        sev = (r.get("severity") or "None").strip()
        tod = (r.get("Time_of_Day") or "?").strip()

        G = load(os.path.join(gt_dir, gt_base), args.gt_scale)
        if G is None:
            continue
        pf = os.path.join(args.pred_dir, map_pred(gt_base, args.name_map))
        P = load(pf, args.pred_scale)
        if P is None:  # fallbacks
            for alt in (gt_base, gt_base.replace("_groundtruth_depth_", "_velodyne_raw_")):
                P = load(os.path.join(args.pred_dir, alt), args.pred_scale)
                if P is not None:
                    break
        if P is None:
            n_missing_pred += 1
            continue
        if P.shape != G.shape:
            P = cv2.resize(P, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)

        sm = std_metrics(P, G)
        if sm is None:
            continue
        for g in groups_of(weather, sev, tod):
            k = gkey(g)
            a = std_acc.setdefault(k, [0, 0.0, 0.0, 0.0, 0.0])
            a[0] += 1
            for j in range(4):
                a[1 + j] += sm[j]

        if do_proposed:
            S_c = load(os.path.join(clean_dir, gt_base.replace("_groundtruth_depth_", "_velodyne_raw_")), args.sparse_scale)
            S_w = load(os.path.join(weather_dir, gt_base.replace("_groundtruth_depth_", "_velodyne_raw_")), args.sparse_scale)
            if S_c is not None and S_w is not None:
                if S_c.shape != G.shape:
                    S_c = cv2.resize(S_c, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)
                if S_w.shape != G.shape:
                    S_w = cv2.resize(S_w, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)
                dm.add(P, G, S_w, S_c, groups=groups_of(weather, sev, tod))
        n_done += 1

    # ---- assemble ordered output ----
    order = ["ALL"] + WEATHER_ORDER \
        + [f"{w}/{s}" for w in WEATHER_ORDER for s in SEV_ORDER] \
        + [f"{w}/{t}" for w in WEATHER_ORDER for t in TOD_ORDER]
    out = {"method": args.name, "set": args.set, "n_scored": n_done,
           "n_missing_pred": n_missing_pred, "n_no_meta": n_no_meta,
           "do_proposed": do_proposed, "tau": args.tau, "groups": {}}
    keys = [k for k in order if k in std_acc]
    for k in keys:
        a = std_acc[k]
        n = a[0]
        rec = {"n": n, "RMSE": a[1] / n, "MAE": a[2] / n,
               "iRMSE": a[3] / n, "iMAE": a[4] / n}
        gk = k if "/" not in k else tuple(k.split("/"))
        if do_proposed and gk in dm.acc:
            pr = dm.reduce(gk)
            rec.update({"ERR": pr["ERR"], "SR": pr["SR"],
                        "DCE": pr["DCE"] * 1000.0, "DCE_mae": pr["DCE_mae"] * 1000.0,
                        "NRR": pr["NRR"], "CPP": pr["CPP"] * 1000.0,
                        "n_cor": pr["n_cor"], "n_drop": pr["n_drop"], "n_keep": pr["n_keep"]})
        out["groups"][k] = rec

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    a = std_acc.get("ALL", [0])
    if a[0]:
        msg = f"[{args.name}/{args.set}] n={a[0]} RMSE={a[1]/a[0]:.1f} MAE={a[2]/a[0]:.1f} iRMSE={a[3]/a[0]:.3f} iMAE={a[4]/a[0]:.3f}"
        if do_proposed and "ALL" in dm.acc:
            pr = dm.reduce("ALL")
            msg += f" | ERR={pr['ERR']:.3f} SR={pr['SR']:.3f} DCE={pr['DCE']*1000:.1f} NRR={pr['NRR']:.3f} CPP={pr['CPP']*1000:.1f}"
        print(msg)
    print(f"  missing_pred={n_missing_pred} no_meta={n_no_meta} -> {args.out}")


if __name__ == "__main__":
    main()
