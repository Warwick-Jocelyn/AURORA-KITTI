"""Per-weather x severity x time-of-day benchmarking for 8 depth methods.

Supports two datasets:
  --dataset aurora  -> val_selection_cropped_TEST + test_small.csv (1600 frames, KITTI naming)
  --dataset dense   -> val_selection_cropped_DENSE + dense_small.csv (1600 frames, scene_*_idx naming)

Outputs into Benchmarking_AURORA/:
  method_<DATASET>_<NAME>_per_image.csv
  method_<DATASET>_<NAME>_breakdown.csv
  comparison_<DATASET>.csv / comparison_<DATASET>.md

For backward compatibility the AURORA dataset *also* writes the legacy filenames
(comparison.csv, comparison.md, method_<NAME>_*.csv) so existing viz/site code keeps
working without changes.
"""
import argparse
import os
import re
import csv
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

ROOT = "/path/to/aurora-root"
OUT_DIR = f"{ROOT}/PhD_Project_2025/Benchmarking_AURORA"

T_VALID = 1e-4
N_WORKERS = 8
METRIC_KEYS = ["RMSE", "MAE", "iRMSE", "iMAE", "REL", "D1", "D2", "D3"]


# ---------- per-dataset config ------------------------------------------------

def _aurora_config():
    test_dir = f"{ROOT}/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST"
    return dict(
        name="aurora",
        test_dir=test_dir,
        csv_path=f"{test_dir}/test_small.csv",
        gt_dir=f"{test_dir}/groundtruth_depth",
        weather_order=["Clear", "Fog", "Rain", "Snow"],
        sev_order=["None", "light", "moderate", "heavy"],
        tod_order=["Day", "Night"],
        # AURORA uses KITTI-drive naming: <drive>_drive_NNNN_sync_image_<cam>_<frame>
        name_parser=_parse_aurora_name,
        # gt filename per (drive, cam, frame)
        gt_filename=lambda drive, cam, frame: f"{drive}_groundtruth_depth_{frame}_{cam}.png",
        # METHODS: per-method dict with dir, suffix, depth_scale, name_kind, optional pred_align.
        # name_kind controls how pred filename is built from (drive, cam, frame).
        methods={
            # H352: BP-Net family now predicts full 352-row depth, no pad_top alignment needed.
            "BPNet":      dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/BP_KITTI_AURORA/selval",
                                suffix=".png", depth_scale=256.0, name_kind="aurora_groundtruth"),
            "DMD3C":      dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/DMD3C_paper_AURORA/selval",
                                suffix=".png", depth_scale=256.0, name_kind="aurora_groundtruth"),
            "DDCD":       dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/DDCD_AURORA/selval",
                                suffix=".png", depth_scale=256.0, name_kind="aurora_groundtruth"),
            "DA3":        dict(dir=f"{test_dir}/da3", suffix=".png", depth_scale=256.0, name_kind="aurora_image"),
            "UniK3D":     dict(dir=f"{ROOT}/Code/UniK3D/results/depth16_test",
                                suffix="_depth_x256.png", depth_scale=256.0, name_kind="aurora_image"),
            "OMNI-DC":    dict(dir=f"{ROOT}/Code/OMNI-DC/output_KITTI_Test_v256",
                                suffix=".png", depth_scale=256.0, name_kind="aurora_image"),
            "PromptDA":   dict(dir=f"{ROOT}/Code/PromptDA/results/Aurora_KITTI_test",
                                suffix=".png", depth_scale=1000.0, name_kind="aurora_image"),
            "Marigold-DC":dict(dir=f"{ROOT}/Code/Marigold-DC/Marigold_DC_result_16bit_Aurora_test",
                                suffix=".png", depth_scale=256.0, name_kind="aurora_velodyne"),
        },
    )


def _dense_config():
    test_dir = f"{ROOT}/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_DENSE"
    return dict(
        name="dense",
        test_dir=test_dir,
        csv_path=f"{test_dir}/dense_small.csv",
        gt_dir=f"{test_dir}/groundtruth_depth",
        weather_order=["Clear", "Fog", "Rain"],   # no Snow in DENSE
        sev_order=["None", "light", "medium", "heavy"],   # DENSE uses "medium" not "moderate"
        tod_order=["Day", "Night"],
        name_parser=_parse_dense_name,
        # DENSE GT filenames are just <stem>.png (no _groundtruth_depth_ infix).
        gt_filename=lambda stem, *_: f"{stem}.png",
        methods={
            "BPNet":      dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/BP_KITTI_DENSE/selval",
                                suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
            "DMD3C":      dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/DMD3C_paper_DENSE/selval",
                                suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
            "DDCD":       dict(dir=f"{ROOT}/PhD_Project_2025/BP-Net/results/DDCD_DENSE/selval",
                                suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
            "DA3":        dict(dir=f"{test_dir}/da3", suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
            "UniK3D":     dict(dir=f"{ROOT}/Code/UniK3D/results/dense_inference/depth16",
                                suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
            "OMNI-DC":    dict(dir=f"{ROOT}/Code/OMNI-DC/output_KITTI_DENSE",
                                suffix=".png", depth_scale=1000.0, name_kind="dense_stem"),
            "PromptDA":   dict(dir=f"{ROOT}/Code/PromptDA/results/DENSE",
                                suffix=".png", depth_scale=1000.0, name_kind="dense_stem"),
            "Marigold-DC":dict(dir=f"{ROOT}/Code/Marigold-DC/Marigold_DC_result_16bit_DENSE",
                                suffix=".png", depth_scale=256.0, name_kind="dense_stem"),
        },
    )


DATASET_CONFIGS = {"aurora": _aurora_config, "dense": _dense_config}


# ---------- name parsers ------------------------------------------------------

def _parse_aurora_name(s):
    """AURORA `name` looks like ..._drive_NNNN_sync_image_<cam>_<frame>."""
    m = re.match(r"^(?P<drive>.*_drive_\d+_sync)_image_(?P<camid>\d+)_(?P<frame>\d+)$", s)
    if not m:
        return None
    return ("aurora", m.group("drive"), f"image_{m.group('camid')}", m.group("frame"))


def _parse_dense_name(s):
    """DENSE `name` is the file stem itself, e.g. scene1_day_clear_0."""
    if not s:
        return None
    return ("dense", s, None, None)


def pred_filename(method_cfg, key):
    suffix = method_cfg["suffix"]
    kind = method_cfg.get("name_kind", "aurora_image")
    if kind == "aurora_groundtruth":
        _, drive, cam, frame = key
        return f"{drive}_groundtruth_depth_{frame}_{cam}{suffix}"
    if kind == "aurora_velodyne":
        _, drive, cam, frame = key
        return f"{drive}_velodyne_raw_{frame}_{cam}{suffix}"
    if kind == "aurora_image":
        _, drive, cam, frame = key
        return f"{drive}_image_{frame}_{cam}{suffix}"
    if kind == "dense_stem":
        _, stem, _, _ = key
        return f"{stem}{suffix}"
    raise ValueError(f"unknown name_kind: {kind}")


def gt_filename_from_key(cfg, key):
    if key[0] == "aurora":
        _, drive, cam, frame = key
        return cfg["gt_filename"](drive, cam, frame)
    else:
        _, stem, _, _ = key
        return cfg["gt_filename"](stem)


# ---------- data loading ------------------------------------------------------

def load_csv_index(cfg):
    out = {}
    with open(cfg["csv_path"]) as f:
        for row in csv.DictReader(f):
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            key = cfg["name_parser"](row["name"])
            if key is None:
                continue
            out[key] = row
    return out


def read_depth_png(path, depth_scale, target_shape=None, align="bilinear"):
    arr = np.array(Image.open(path)).astype(np.float32) / depth_scale
    if target_shape is not None and arr.shape != target_shape:
        H, W = target_shape
        if align == "pad_top" and arr.shape[1] == W and arr.shape[0] < H:
            # BP-Net only predicts the bottom 256 rows; the top H-256 rows have no
            # prediction. Fill with NaN (not 0) so per_image_metrics can mask them
            # out — otherwise pred=0 vs real GT inflates RMSE on datasets like DENSE
            # where GT is dense in those rows.
            padded = np.full((H, W), np.nan, dtype=np.float32)
            padded[H - arr.shape[0]:, :] = arr
            arr = padded
        else:
            arr = np.array(Image.fromarray(arr, mode="F").resize((W, H), Image.BILINEAR), dtype=np.float32)
    return arr


def per_image_metrics(pred, gt, crop_top_rows=0):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    # Mask: GT must be valid AND pred must be finite (NaN = "method did not predict here").
    mask = (gt > T_VALID) & np.isfinite(pred)
    if crop_top_rows > 0:
        # Standard KITTI eval protocol: lidar GT is concentrated in the bottom 256 rows;
        # zero-mask the top N rows uniformly across all methods.
        mask[:crop_top_rows, :] = False
    n_valid = max(int(mask.sum()), 1)
    # Replace NaN preds with 0 in the masked-out area so vector ops stay finite.
    pred = np.where(np.isfinite(pred), pred, 0.0)
    diff = (pred - gt) * mask
    pred_inv = 1.0 / (pred + 1e-8)
    gt_inv = 1.0 / (gt + 1e-8)
    diff_inv = (pred_inv - gt_inv) * mask
    rmse = float(np.sqrt(np.sum(diff ** 2) / n_valid))
    mae = float(np.sum(np.abs(diff)) / n_valid)
    irmse = float(np.sqrt(np.sum(diff_inv ** 2) / n_valid))
    imae = float(np.sum(np.abs(diff_inv)) / n_valid)
    rel = float(np.sum(np.abs(diff) / (gt * mask + 1e-8)) / n_valid)
    safe_pred = np.where(mask, pred, 1.0)
    safe_gt = np.where(mask, gt, 1.0)
    ratio = np.maximum(safe_gt / (safe_pred + 1e-8), safe_pred / (safe_gt + 1e-8))
    ratio = np.where(mask, ratio, 10000.0)
    d1 = float(np.sum(ratio < 1.25) / n_valid)
    d2 = float(np.sum(ratio < 1.25 ** 2) / n_valid)
    d3 = float(np.sum(ratio < 1.25 ** 3) / n_valid)
    return dict(RMSE=rmse, MAE=mae, iRMSE=irmse, iMAE=imae, REL=rel, D1=d1, D2=d2, D3=d3,
                n_valid=n_valid)


# ---------- module-level globals so worker can pickle method dicts ------------
_WORKER_CFG = None
_WORKER_METHOD = None
_WORKER_METHOD_CFG = None
_WORKER_CROP_TOP = 0


def _worker_init(cfg, method, method_cfg, crop_top):
    global _WORKER_CFG, _WORKER_METHOD, _WORKER_METHOD_CFG, _WORKER_CROP_TOP
    _WORKER_CFG = cfg
    _WORKER_METHOD = method
    _WORKER_METHOD_CFG = method_cfg
    _WORKER_CROP_TOP = crop_top


def _worker(task):
    key, meta = task
    method_cfg = _WORKER_METHOD_CFG
    cfg = _WORKER_CFG
    pred_path = os.path.join(method_cfg["dir"], pred_filename(method_cfg, key))
    gt_path = os.path.join(cfg["gt_dir"], gt_filename_from_key(cfg, key))
    if not os.path.isfile(pred_path):
        return key, dict(error=f"missing pred: {pred_path}")
    if not os.path.isfile(gt_path):
        return key, dict(error=f"missing gt: {gt_path}")
    gt = read_depth_png(gt_path, 256.0)
    pred = read_depth_png(pred_path, method_cfg["depth_scale"], target_shape=gt.shape,
                          align=method_cfg.get("pred_align", "bilinear"))
    if pred.shape != gt.shape:
        return key, dict(error=f"shape mismatch pred {pred.shape} vs gt {gt.shape}")
    m = per_image_metrics(pred, gt, crop_top_rows=_WORKER_CROP_TOP)
    m.update(weather=meta.get("weather", "?") or "?",
             severity=meta.get("severity", "?") or "?",
             tod=meta.get("Time_of_Day", "?") or "?",
             scenario=meta.get("Scenario", "?") or "?",
             key="|".join(str(k) for k in key))
    return key, m


def eval_method(method, method_cfg, csv_idx, cfg, crop_top=0):
    print(f"[{cfg['name']}/{method}] eval start, frames={len(csv_idx)}, dir={method_cfg['dir']}, crop_top={crop_top}")
    if not os.path.isdir(method_cfg["dir"]):
        print(f"  WARNING: dir does not exist, skipping {method}")
        return []
    t0 = time.time()
    tasks = list(csv_idx.items())
    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init,
                             initargs=(cfg, method, method_cfg, crop_top)) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            key, m = fut.result()
            if "error" in m:
                errors.append((key, m["error"]))
            else:
                rows.append(m)
            if (i + 1) % 200 == 0:
                print(f"  [{cfg['name']}/{method}] {i+1}/{len(futs)} ({time.time()-t0:.0f}s)")
    print(f"[{cfg['name']}/{method}] done in {time.time()-t0:.0f}s, ok={len(rows)}, err={len(errors)}")
    if errors:
        for k, e in errors[:3]:
            print(f"  err {k}: {e}")
    return rows


def agg(rows_subset):
    if not rows_subset:
        return None
    out = {"N": len(rows_subset)}
    for k in METRIC_KEYS:
        out[k] = float(np.mean([r[k] for r in rows_subset]))
    return out


def write_breakdown_csv(rows, path, cfg):
    by_w = defaultdict(list); by_ws = defaultdict(list); by_wt = defaultdict(list)
    by_wst = defaultdict(list); by_t = defaultdict(list)
    for r in rows:
        w, s, t = r["weather"], r["severity"], r["tod"]
        by_w[w].append(r); by_ws[(w, s)].append(r); by_wt[(w, t)].append(r)
        by_wst[(w, s, t)].append(r); by_t[t].append(r)

    csv_rows = [["group_type", "weather", "severity", "tod", "N",
                 "RMSE_mm", "MAE_mm", "iRMSE_1km", "iMAE_1km",
                 "REL", "D1", "D2", "D3"]]

    def push(gt, w, s, t, m):
        if m is None: return
        csv_rows.append([
            gt, w, s, t, m["N"],
            f"{m['RMSE']*1000:.2f}", f"{m['MAE']*1000:.2f}",
            f"{m['iRMSE']*1000:.2f}", f"{m['iMAE']*1000:.2f}",
            f"{m['REL']:.4f}", f"{m['D1']:.4f}", f"{m['D2']:.4f}", f"{m['D3']:.4f}",
        ])

    push("overall", "ALL", "ALL", "ALL", agg(rows))
    for t in cfg["tod_order"]:
        push("tod", "ALL", "ALL", t, agg(by_t.get(t, [])))
    for w in cfg["weather_order"]:
        push("weather", w, "ALL", "ALL", agg(by_w.get(w, [])))
    for w in cfg["weather_order"]:
        for s in cfg["sev_order"]:
            if (w, s) in by_ws:
                push("weather+severity", w, s, "ALL", agg(by_ws[(w, s)]))
    for w in cfg["weather_order"]:
        for t in cfg["tod_order"]:
            if (w, t) in by_wt:
                push("weather+tod", w, "ALL", t, agg(by_wt[(w, t)]))
    for w in cfg["weather_order"]:
        for s in cfg["sev_order"]:
            for t in cfg["tod_order"]:
                if (w, s, t) in by_wst:
                    push("weather+severity+tod", w, s, t, agg(by_wst[(w, s, t)]))

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    return csv_rows


def write_per_image_csv(rows, path):
    cols = ["key", "weather", "severity", "tod", "scenario", "n_valid"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def run_one_dataset(cfg, crop_top=0):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_idx = load_csv_index(cfg)
    print(f"[{cfg['name']}] loaded {len(csv_idx)} csv rows from {cfg['csv_path']} (crop_top={crop_top})")

    method_breakdowns = {}
    method_order = list(cfg["methods"].keys())
    for method in method_order:
        method_cfg = cfg["methods"][method]
        rows = eval_method(method, method_cfg, csv_idx, cfg, crop_top=crop_top)
        per_img_path = os.path.join(OUT_DIR, f"method_{cfg['name']}_{method}_per_image.csv")
        write_per_image_csv(rows, per_img_path)
        bd_path = os.path.join(OUT_DIR, f"method_{cfg['name']}_{method}_breakdown.csv")
        bd_csv_rows = write_breakdown_csv(rows, bd_path, cfg)
        # Backward-compat: AURORA also writes legacy filenames
        if cfg["name"] == "aurora":
            write_per_image_csv(rows, os.path.join(OUT_DIR, f"method_{method}_per_image.csv"))
            write_breakdown_csv(rows, os.path.join(OUT_DIR, f"method_{method}_breakdown.csv"), cfg)
        header = bd_csv_rows[0]
        method_breakdowns[method] = [dict(zip(header, r)) for r in bd_csv_rows[1:]]
        print(f"  wrote {bd_path}")

    # ---------- comparison.csv ----------
    rows_long = [["method", "group_type", "weather", "severity", "tod", "N",
                  "RMSE_mm", "MAE_mm", "iRMSE_1km", "iMAE_1km",
                  "REL", "D1", "D2", "D3"]]
    for m in method_order:
        for r in method_breakdowns[m]:
            rows_long.append([m, r["group_type"], r["weather"], r["severity"], r["tod"],
                              r["N"], r["RMSE_mm"], r["MAE_mm"], r["iRMSE_1km"],
                              r["iMAE_1km"], r["REL"], r["D1"], r["D2"], r["D3"]])
    comp_csv = os.path.join(OUT_DIR, f"comparison_{cfg['name']}.csv")
    with open(comp_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows_long)
    print(f"wrote {comp_csv}")
    if cfg["name"] == "aurora":
        with open(os.path.join(OUT_DIR, "comparison.csv"), "w", newline="") as f:
            csv.writer(f).writerows(rows_long)

    # ---------- comparison.md ----------
    idx = {(m, (r["group_type"], r["weather"], r["severity"], r["tod"])): r
           for m in method_order for r in method_breakdowns[m]}

    metric_tables = [
        ("RMSE_mm", "RMSE (mm) — lower is better"),
        ("MAE_mm", "MAE (mm) — lower is better"),
        ("REL", "REL — lower is better"),
        ("D1", "D^1 (δ < 1.25) — higher is better"),
        ("iRMSE_1km", "iRMSE (1/km) — lower is better"),
        ("iMAE_1km", "iMAE (1/km) — lower is better"),
    ]

    md = []
    md.append(f"# {cfg['name'].upper()}-KITTI per-weather benchmark — {len(method_order)} methods\n")
    md.append(f"Frames: {len(csv_idx)} (from `{os.path.basename(cfg['csv_path'])}`)\n")
    md.append(f"Methods order: {', '.join(method_order)}\n")
    md.append("Units: RMSE/MAE in **mm**; iRMSE/iMAE in **1/km**; REL/D^k unitless. Lower is better for RMSE/MAE/iRMSE/iMAE/REL; higher is better for D^k.\n")
    cost_path = os.path.join(OUT_DIR, "method_cost.md")
    if os.path.isfile(cost_path) and cfg["name"] == "aurora":
        with open(cost_path) as f: md.append(f.read())
        md.append("")

    def collect_groups():
        items = [("overall", "ALL", "ALL", "ALL", "ALL")]
        for t in cfg["tod_order"]:
            items.append((f"All / {t}", "tod", "ALL", "ALL", t))
        for w in cfg["weather_order"]:
            items.append((f"{w}", "weather", w, "ALL", "ALL"))
        for w in cfg["weather_order"]:
            for s in cfg["sev_order"]:
                items.append((f"{w} / {s}", "weather+severity", w, s, "ALL"))
        for w in cfg["weather_order"]:
            for t in cfg["tod_order"]:
                items.append((f"{w} / {t}", "weather+tod", w, "ALL", t))
        for w in cfg["weather_order"]:
            for s in cfg["sev_order"]:
                for t in cfg["tod_order"]:
                    items.append((f"{w} / {s} / {t}", "weather+severity+tod", w, s, t))
        return items

    def best_idx(values, lower_is_better):
        best, bi = None, -1
        for i, v in enumerate(values):
            if v is None: continue
            if best is None or (lower_is_better and v < best) or (not lower_is_better and v > best):
                best, bi = v, i
        return bi

    for col_key, title in metric_tables:
        md.append(f"\n## {title}\n")
        header = ["Group", "N"] + method_order
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        lower_is_better = col_key not in ("D1", "D2", "D3")
        for label, gt, w, s, t in collect_groups():
            n_val = "-"; vals = []
            for m in method_order:
                r = idx.get((m, (gt, w, s, t)))
                if r is None: vals.append(None)
                else:
                    if n_val == "-": n_val = r["N"]
                    try: vals.append(float(r[col_key]))
                    except (ValueError, KeyError): vals.append(None)
            if all(v is None for v in vals): continue
            bi = best_idx(vals, lower_is_better)
            cells = []
            for i, v in enumerate(vals):
                if v is None: cells.append("—")
                else:
                    if col_key in ("REL", "D1", "D2", "D3"): s_ = f"{v:.4f}"
                    else: s_ = f"{v:.2f}"
                    if i == bi: s_ = f"**{s_}**"
                    cells.append(s_)
            md.append("| " + " | ".join([label, str(n_val)] + cells) + " |")

    out_md = os.path.join(OUT_DIR, f"comparison_{cfg['name']}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {out_md}")
    if cfg["name"] == "aurora":
        with open(os.path.join(OUT_DIR, "comparison.md"), "w") as f:
            f.write("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="aurora")
    p.add_argument("--metric_crop_top", type=int, default=0,
                   help="Mask out the top N rows of GT before computing metrics. "
                        "Use 96 to evaluate only on the bottom 256 rows (KITTI lidar-GT region) "
                        "for a fairer comparison when some methods only predict the bottom region.")
    args = p.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]()
    run_one_dataset(cfg, crop_top=args.metric_crop_top)


if __name__ == "__main__":
    main()
