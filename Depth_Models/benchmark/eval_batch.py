#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-process batch evaluator (AV-aware). On this host an on-access AV scanner
makes every cv2.imread/.so-open cost ~1-2s, so we:
  (1) import torch/criteria ONCE,
  (2) load shared GT/clean/weather per split ONCE into RAM (threaded reads via
      open().read()+imdecode, which is ~6x faster than cv2.imread),
  (3) evaluate every method (threaded pred reads), writing json/<m>__<SET>.json.

Reads a TSV: name<TAB>set<TAB>pred_dir<TAB>name_map<TAB>scale (lines, '#' to skip).
Skips a job if its JSON exists (unless --force) or its pred_dir has <min_frames PNGs.
Run in tmux (no timeout). Re-run anytime to pick up newly-finished methods.
"""
import os, sys, csv, json, argparse, time, glob, numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor

DMD3C = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C"
sys.path.insert(0, DMD3C)
from criteria import DenoiseMetrics  # noqa: E402

KBASE = os.path.join(DMD3C, "datas", "kitti")
RZB = os.path.dirname(os.path.abspath(__file__))
SETS = {"TEST": ("val_selection_cropped_TEST", "test_small.csv"),
        "DENSE": ("val_selection_cropped_DENSE", "dense_small.csv")}
MAXD = 80.0
WEATHER_ORDER = ["Clear", "Fog", "Rain", "Snow"]
SEV_ORDER = ["light", "moderate", "heavy"]
TOD_ORDER = ["Day", "Night"]
NWORK = 24


def _read_raw(path):
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
    return img


def threaded_load(paths, scale):
    """Return dict idx->float32 array (metres). None entries for missing."""
    out = [None] * len(paths)
    def job(i):
        a = _read_raw(paths[i])
        out[i] = None if a is None else (a.astype(np.float32) / scale)
    with ThreadPoolExecutor(max_workers=NWORK) as ex:
        list(ex.map(job, range(len(paths))))
    return out


def map_pred(gt_base, name_map):
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
    Pm = P[m]
    p = np.clip(Pm, 1e-3, None).astype(np.float64)   # linear metrics: unchanged
    g = G[m].astype(np.float64)
    # Inverse-depth metrics: exclude HOLE pixels (pred<=0, i.e. the method output
    # nothing there) instead of clipping them to 1mm. Clipping maps a hole to
    # 1/0.001 = 1000 (1/m), which detonates iRMSE/iMAE for methods with even a
    # handful of unfilled pixels (e.g. IP-Basic). This is a NO-OP for dense
    # predictions (no holes) and keeps all methods on an identical rule.
    vp = Pm > 1e-3
    if vp.sum() > 0:
        gi = 1.0 / g[vp]; pi = 1.0 / p[vp]
        irmse = float(np.sqrt(np.mean((pi - gi) ** 2)) * 1000.0)
        imae = float(np.mean(np.abs(pi - gi)) * 1000.0)
    else:
        irmse = imae = float("nan")
    return (np.sqrt(np.mean((p - g) ** 2)) * 1000.0,
            np.mean(np.abs(p - g)) * 1000.0,
            irmse, imae)


def groups_of(w, s, t):
    return ["ALL", w, (w, s), (w, t)]


def gkey(g):
    return g if isinstance(g, str) else "/".join(g)


class Shared:
    """Per-split shared tensors loaded once."""
    def __init__(self, set_):
        self.set_ = set_
        sub, csvn = SETS[set_]
        root = os.path.join(KBASE, sub)
        self.gt_dir = os.path.join(root, "groundtruth_depth")
        self.gt_files = sorted(f for f in os.listdir(self.gt_dir) if f.endswith(".png"))
        # meta keyed by GT basename
        self.meta = {}
        rows = list(csv.DictReader(open(os.path.join(root, csvn))))
        if set_ == "TEST":
            for r in rows:
                nm = r["name"].strip(); drive, rest = nm.split("_image_"); cam, frame = rest.split("_")
                self.meta[f"{drive}_groundtruth_depth_{frame}_image_{cam}.png"] = r
        else:
            for r in rows:
                self.meta[r["name"].strip() + ".png"] = r
        # filter gt_files to those with meta
        self.gt_files = [g for g in self.gt_files if g in self.meta]
        t = time.time()
        self.G = threaded_load([os.path.join(self.gt_dir, g) for g in self.gt_files], 256.0)
        self.do_prop = (set_ == "TEST") and os.path.isdir(os.path.join(root, "velodyne_raw_clean"))
        if self.do_prop:
            cd = os.path.join(root, "velodyne_raw_clean"); wd = os.path.join(root, "velodyne_raw")
            vel = [g.replace("_groundtruth_depth_", "_velodyne_raw_") for g in self.gt_files]
            self.Sc = threaded_load([os.path.join(cd, v) for v in vel], 255.0)
            self.Sw = threaded_load([os.path.join(wd, v) for v in vel], 255.0)
        print(f"[shared {set_}] loaded {len(self.gt_files)} frames, proposed={self.do_prop} in {time.time()-t:.0f}s", flush=True)


def eval_method(sh, name, pred_dir, name_map, scale, out, crop_gt_top: int = 0):
    t = time.time()
    pred_paths = [os.path.join(pred_dir, map_pred(g, name_map)) for g in sh.gt_files]
    P = threaded_load(pred_paths, scale)
    std_acc = {}
    dm = DenoiseMetrics(tau=0.05) if sh.do_prop else None
    n_done = n_missing = 0
    for i, g in enumerate(sh.gt_files):
        G = sh.G[i]; Pi = P[i]
        if G is None or Pi is None:
            n_missing += 1; continue
        # Bottom-crop protocol (BP-Net selval, height=256): keep the bottom
        # (H - crop_gt_top) rows of BOTH pred and GT, so every method is scored
        # on exactly the same rows. We CROP the pred (never resize) when it is
        # full-height; a pred already delivered at the cropped height is kept.
        if crop_gt_top > 0:
            G = G[crop_gt_top:, :]
            if Pi.shape[0] == G.shape[0] + crop_gt_top:
                Pi = Pi[crop_gt_top:, :]
        if Pi.shape != G.shape:
            # Shape mismatch: resize pred to GT size. Correct approach is to
            # generate preds at native GT resolution (352x1216 for DENSE) so
            # this path is never triggered for the benchmark methods.
            Pi = cv2.resize(Pi, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)
        sm = std_metrics(Pi, G)
        if sm is None:
            continue
        r = sh.meta[g]
        w = (r.get("weather") or "?").strip(); s = (r.get("severity") or "None").strip(); td = (r.get("Time_of_Day") or "?").strip()
        for grp in groups_of(w, s, td):
            k = gkey(grp); a = std_acc.setdefault(k, [0, 0.0, 0.0, 0.0, 0.0]); a[0] += 1
            for j in range(4):
                a[1 + j] += sm[j]
        if sh.do_prop:
            Sc = sh.Sc[i]; Sw = sh.Sw[i]
            if Sc is not None and Sw is not None:
                dm.add(Pi, G, Sw, Sc, groups=groups_of(w, s, td))
        n_done += 1
    order = ["ALL"] + WEATHER_ORDER + [f"{w}/{s}" for w in WEATHER_ORDER for s in SEV_ORDER] + [f"{w}/{t}" for w in WEATHER_ORDER for t in TOD_ORDER]
    proto = f"crop_gt_top={crop_gt_top}" if crop_gt_top else "full_gt"
    res = {"method": name, "set": sh.set_, "n_scored": n_done, "n_missing_pred": n_missing,
           "do_proposed": sh.do_prop, "eval_protocol": proto, "groups": {}}
    for k in [k for k in order if k in std_acc]:
        a = std_acc[k]; n = a[0]
        rec = {"n": n, "RMSE": a[1]/n, "MAE": a[2]/n, "iRMSE": a[3]/n, "iMAE": a[4]/n}
        gk = k if "/" not in k else tuple(k.split("/"))
        if sh.do_prop and gk in dm.acc:
            pr = dm.reduce(gk)
            rec.update({"ERR": pr["ERR"], "SR": pr["SR"], "DCE": pr["DCE"]*1000.0,
                        "DCE_mae": pr["DCE_mae"]*1000.0, "NRR": pr["NRR"], "CPP": pr["CPP"]*1000.0,
                        "n_cor": pr["n_cor"], "n_drop": pr["n_drop"], "n_keep": pr["n_keep"]})
        res["groups"][k] = rec
    json.dump(res, open(out, "w"), indent=2)
    a = std_acc.get("ALL", [0])
    msg = f"[{name}/{sh.set_}] n={a[0]} miss={n_missing} RMSE={a[1]/a[0]:.1f} MAE={a[2]/a[0]:.1f}" if a[0] else f"[{name}/{sh.set_}] NO DATA miss={n_missing}"
    if a[0] and sh.do_prop and "ALL" in dm.acc:
        pr = dm.reduce("ALL"); msg += f" ERR={pr['ERR']:.3f} DCE={pr['DCE']*1000:.0f}"
    print(msg + f"  ({time.time()-t:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=os.path.join(RZB, "jobs_all.tsv"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min_frames", type=int, default=1500)
    ap.add_argument("--jsondir", default="json",
                    help="output subdir under this script's dir (e.g. json, json_full352)")
    ap.add_argument("--zero_crop", action="store_true",
                    help="ignore the per-row crop_gt_top column (full-height eval, for the 352 appendix)")
    args = ap.parse_args()
    JOUT = os.path.join(RZB, args.jsondir)
    jobs = []
    for line in open(args.jobs):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        # 6th column (optional): crop_gt_top — number of rows to drop from top of GT
        # before eval. Used for methods evaluated at h<352 (e.g. DDCD DENSE h=256,
        # where crop_gt_top=96 aligns GT bottom-256 rows with the 256-height pred).
        crop_gt_top = int(parts[5]) if len(parts) >= 6 and parts[5].strip().lstrip("-").isdigit() else 0
        if args.zero_crop:
            crop_gt_top = 0
        jobs.append(parts[:5] + [crop_gt_top])
    os.makedirs(JOUT, exist_ok=True)
    # group by set so shared loads once
    for set_ in ("TEST", "DENSE"):
        todo = []
        for name, s, pdir, nmap, scale, crop_gt_top in jobs:
            if s != set_:
                continue
            out = os.path.join(JOUT, f"{name}__{set_}.json")
            if os.path.exists(out) and not args.force:
                continue
            npng = len(glob.glob(os.path.join(pdir, "*.png"))) if os.path.isdir(pdir) else 0
            if npng < args.min_frames:
                print(f"[skip] {name}/{set_}: pred_dir has {npng} pngs (<{args.min_frames}) {pdir}", flush=True)
                continue
            todo.append((name, pdir, nmap, float(scale), int(crop_gt_top)))
        if not todo:
            print(f"[{set_}] nothing to do", flush=True)
            continue
        print(f"[{set_}] {len(todo)} methods to eval: {[t[0] for t in todo]}", flush=True)
        sh = Shared(set_)
        for name, pdir, nmap, scale, crop_gt_top in todo:
            try:
                eval_method(sh, name, pdir, nmap, scale, os.path.join(JOUT, f"{name}__{set_}.json"),
                            crop_gt_top=crop_gt_top)
            except Exception as e:
                print(f"[ERROR] {name}/{set_}: {e}", flush=True)
        del sh
    print("BATCH EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
