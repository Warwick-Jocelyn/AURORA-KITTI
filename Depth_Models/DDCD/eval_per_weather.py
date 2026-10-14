"""
Per-image evaluation of the rank-01 checkpoint on val_selection_cropped_TEST,
broken down by weather x severity x time_of_day from test_small.csv.

Outputs:
  weather_breakdown.csv   - one row per (weather, severity, tod) combo
  weather_breakdown.md    - same data formatted as markdown tables
"""
import os
import re
import sys
import csv
import time
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf

_THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _THIS)
sys.path.append(os.path.abspath(os.path.join(_THIS, "..")))

from utils import Trainer
from test import _pick_inputs_for_forward

CKPT = "checkpoints/_unified/ddcd_aurora.pth"
CSV_PATH = "datas/kitti/val_selection_cropped_TEST/test_small.csv"
OUT_CSV = "weather_breakdown.csv"
OUT_MD = "weather_breakdown.md"


def parse_pred_name(fname: str):
    """Image basename in TEST: <drive>_groundtruth_depth_<frame>_<cam>.png
       drive  = '2011_09_26_drive_0011_sync'
       cam    = 'image_02'
       frame  = '0000000005'
    """
    stem = os.path.splitext(fname)[0]
    m = re.match(r"^(?P<drive>.*_drive_\d+_sync)_(image|groundtruth_depth|velodyne_raw)_"
                 r"(?P<frame>\d+)_(?P<cam>image_\d+)$", stem)
    if not m:
        return None
    return (m.group("drive"), m.group("cam"), m.group("frame"))


def parse_csv_name(s: str):
    """CSV `name` field: <drive>_image_<camid>_<frame>, where camid is e.g. '02' or '03'."""
    m = re.match(r"^(?P<drive>.*_drive_\d+_sync)_image_(?P<camid>\d+)_(?P<frame>\d+)$", s)
    if not m:
        return None
    return (m.group("drive"), f"image_{m.group('camid')}", m.group("frame"))


def load_csv_index(path):
    """Return: key (drive, cam, frame) -> row dict."""
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            key = parse_csv_name(row["name"])
            if key is None:
                continue
            out[key] = row
    return out


def per_image_metrics(pred, gt, max_depth=80.0, t_valid=1e-4):
    """Compute per-image RMSE/MAE/iRMSE/iMAE/REL/D^k matching criteria.MetricALL.
    pred, gt: (B,1,H,W) torch tensors in metres.
    Returns list of dicts of length B.
    """
    pred = pred.detach()
    gt = gt.detach()
    pred_inv = 1.0 / (pred + 1e-8)
    gt_inv = 1.0 / (gt + 1e-8)
    mask = (gt > t_valid).float()
    B = pred.shape[0]
    out = []
    for b in range(B):
        m = mask[b]
        n_valid = m.sum().clamp_min(1.0)
        diff = (pred[b] - gt[b]) * m
        diff_inv = (pred_inv[b] - gt_inv[b]) * m
        rmse = torch.sqrt((diff ** 2).sum() / n_valid).item()
        mae = (diff.abs().sum() / n_valid).item()
        irmse = torch.sqrt((diff_inv ** 2).sum() / n_valid).item()
        imae = (diff_inv.abs().sum() / n_valid).item()
        rel = (diff.abs() / (gt[b] * m + 1e-8)).sum() / n_valid
        rel = rel.item()
        # delta accuracies
        ratio = torch.max(gt[b] * m / (pred[b] * m + 1e-8),
                          pred[b] * m / (gt[b] * m + 1e-8))
        ratio = torch.where(m > 0, ratio, torch.full_like(ratio, 10000.0))
        d1 = (ratio < 1.25).float().sum() / n_valid
        d2 = (ratio < 1.25 ** 2).float().sum() / n_valid
        d3 = (ratio < 1.25 ** 3).float().sum() / n_valid
        out.append(dict(
            RMSE=rmse, MAE=mae, iRMSE=irmse, iMAE=imae, REL=rel,
            D1=d1.item(), D2=d2.item(), D3=d3.item(),
            n_valid=n_valid.item(),
        ))
    return out


def main():
    csv_idx = load_csv_index(CSV_PATH)
    print(f"loaded {len(csv_idx)} CSV rows from {CSV_PATH}")

    with initialize(version_base="1.2", config_path="configs"):
        cfg = compose(config_name="config", overrides=[
            "metric=MetricALL", "data=KITTI", "data.testset.mode=selval",
            "test_batch_size=1", "num_workers=1", "name=PER_WEATHER", "gpus=[0]",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.chpt = CKPT

    rows = []  # list of dict: name + metrics + weather/severity/tod
    miss_meta = 0
    t0 = time.time()
    with Trainer(cfg) as run:
        net = run.net_ema.module
        net.eval()
        names = run.testloader.dataset.names  # depth filenames
        idx = 0
        with torch.no_grad():
            for datas in tqdm(run.testloader, desc="eval", dynamic_ncols=True):
                datas = run.init_cuda(*datas)
                inputs = _pick_inputs_for_forward(datas, net, run=None)
                out = net(*inputs)
                pred = out[-1] if isinstance(out, (list, tuple)) else out
                gt = datas[-1]
                metrics_per = per_image_metrics(pred, gt)
                for b, mt in enumerate(metrics_per):
                    fname = names[idx]
                    idx += 1
                    key = parse_pred_name(fname)
                    meta = csv_idx.get(key) if key else None
                    if meta is None:
                        miss_meta += 1
                        weather, severity, tod, scenario = "?", "?", "?", "?"
                    else:
                        weather = meta.get("weather", "?") or "?"
                        severity = meta.get("severity", "?") or "?"
                        tod = meta.get("Time_of_Day", "?") or "?"
                        scenario = meta.get("Scenario", "?") or "?"
                    rows.append(dict(
                        name=fname, weather=weather, severity=severity,
                        tod=tod, scenario=scenario, **mt,
                    ))
    print(f"eval finished in {time.time()-t0:.0f}s; rows={len(rows)}, missing meta={miss_meta}")

    # ---------------- Aggregations ----------------
    metric_keys = ["RMSE", "MAE", "iRMSE", "iMAE", "REL", "D1", "D2", "D3"]
    SCALE_x1000 = {"RMSE", "MAE", "iRMSE", "iMAE"}

    def agg(rows_subset):
        if not rows_subset:
            return None
        n = len(rows_subset)
        out = {"N": n}
        # weighted by n_valid for RMSE-style metrics? KITTI convention: per-image mean.
        for k in metric_keys:
            vals = [r[k] for r in rows_subset]
            v = float(np.mean(vals))
            out[k] = v
        return out

    def fmt(name, v):
        if v is None:
            return "—"
        if name in SCALE_x1000:
            return f"{v * 1000:.2f}"
        return f"{v:.4f}"

    # group: by weather, by weather+severity, by weather+tod, by weather+severity+tod,
    # plus by tod (all-weather daytime / nighttime)
    by_w = defaultdict(list)
    by_ws = defaultdict(list)
    by_wt = defaultdict(list)
    by_wst = defaultdict(list)
    by_t = defaultdict(list)
    overall = []
    for r in rows:
        overall.append(r)
        w, s, t = r["weather"], r["severity"], r["tod"]
        by_w[w].append(r)
        by_ws[(w, s)].append(r)
        by_wt[(w, t)].append(r)
        by_wst[(w, s, t)].append(r)
        by_t[t].append(r)

    weather_order = ["Clear", "Fog", "Rain", "Snow"]
    sev_order = ["None", "light", "moderate", "heavy"]
    tod_order = ["Day", "Night"]

    # ------- write CSV -------
    csv_rows = []
    csv_rows.append(["group_type", "weather", "severity", "tod", "N",
                     "RMSE_mm", "MAE_mm", "iRMSE_1km", "iMAE_1km",
                     "REL", "D1", "D2", "D3"])
    def push(group_type, w, s, t, m):
        if m is None:
            return
        csv_rows.append([
            group_type, w, s, t, m["N"],
            f"{m['RMSE']*1000:.2f}", f"{m['MAE']*1000:.2f}",
            f"{m['iRMSE']*1000:.2f}", f"{m['iMAE']*1000:.2f}",
            f"{m['REL']:.4f}", f"{m['D1']:.4f}", f"{m['D2']:.4f}", f"{m['D3']:.4f}",
        ])

    push("overall", "ALL", "ALL", "ALL", agg(overall))
    for t in tod_order:
        push("tod", "ALL", "ALL", t, agg(by_t.get(t, [])))
    for w in weather_order:
        push("weather", w, "ALL", "ALL", agg(by_w.get(w, [])))
    for w in weather_order:
        for s in sev_order:
            if (w, s) in by_ws:
                push("weather+severity", w, s, "ALL", agg(by_ws[(w, s)]))
    for w in weather_order:
        for t in tod_order:
            if (w, t) in by_wt:
                push("weather+tod", w, "ALL", t, agg(by_wt[(w, t)]))
    for w in weather_order:
        for s in sev_order:
            for t in tod_order:
                if (w, s, t) in by_wst:
                    push("weather+severity+tod", w, s, t, agg(by_wst[(w, s, t)]))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)
    print(f"wrote {OUT_CSV}")

    # ------- write Markdown -------
    md = []
    md.append("# Rank-01 checkpoint — TEST set per-weather breakdown\n")
    md.append(f"Checkpoint: `{CKPT}`")
    md.append(f"CSV: `{CSV_PATH}` ({len(csv_idx)} rows, "
              f"{len(rows)-miss_meta}/{len(rows)} matched)\n")
    md.append("Units: RMSE/MAE in **mm** (raw × 1000, 2 dec); iRMSE/iMAE in **1/km** (raw × 1000, 2 dec); REL/Dk raw 4 dec.\n")

    def render_table(title, items):
        md.append(f"## {title}\n")
        md.append("| Group | N | RMSE (mm) | MAE (mm) | iRMSE (1/km) | iMAE (1/km) | REL | D^1 | D^2 | D^3 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, m in items:
            if m is None:
                continue
            md.append(f"| {label} | {m['N']} | {m['RMSE']*1000:.2f} | "
                      f"{m['MAE']*1000:.2f} | {m['iRMSE']*1000:.2f} | "
                      f"{m['iMAE']*1000:.2f} | {m['REL']:.4f} | "
                      f"{m['D1']:.4f} | {m['D2']:.4f} | {m['D3']:.4f} |")
        md.append("")

    render_table("Overall + by Time_of_Day + by weather",
                 [("ALL", agg(overall)),
                  ("All weathers / Day (daytime)", agg(by_t.get("Day", []))),
                  ("All weathers / Night (nighttime)", agg(by_t.get("Night", [])))]
                 + [(w, agg(by_w.get(w, []))) for w in weather_order])
    render_table("By weather + severity",
                 [(f"{w} / {s}", agg(by_ws.get((w, s), [])))
                  for w in weather_order for s in sev_order if (w, s) in by_ws])
    render_table("By weather + Time_of_Day",
                 [(f"{w} / {t}", agg(by_wt.get((w, t), [])))
                  for w in weather_order for t in tod_order if (w, t) in by_wt])
    render_table("By weather + severity + Time_of_Day",
                 [(f"{w} / {s} / {t}", agg(by_wst.get((w, s, t), [])))
                  for w in weather_order for s in sev_order for t in tod_order
                  if (w, s, t) in by_wst])

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
