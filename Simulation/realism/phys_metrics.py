#!/usr/bin/env python3
"""
Our own physically-grounded weather-realism metrics (complements FID).

For every image we compute physically interpretable descriptors that adverse
weather is known to change (Koschmieder / dark-channel theory + streak/particle
statistics):
  dcp        Dark-Channel-Prior mean   (fog/haze lifts the dark channel  -> up)
  contrast   global luminance std       (fog attenuates contrast          -> down)
  airlight   mean of brightest 0.1%     (fog/veiling airlight             -> up)
  sat        mean HSV saturation        (haze desaturates                 -> down)
  transmit   Koschmieder transmission   (fog lowers scene transmission    -> down)
  hf         Laplacian variance         (rain/snow add HF; fog removes it)
  colorful   Hasler-Susstrunk           (weather shifts colour spread)
  bright     grey mean brightness
  spots      white top-hat blob density (lens raindrops/snowflakes         -> up)

Realism test: for each REAL target weather set we compute, per feature, the
1-D Wasserstein distance W1(AURORA-<w>, real-<w>) and compare it to
W1(AURORA-clear, real-<w>). If our matched synthetic weather is distributionally
closest to the real weather (and closer than clear), the rendering is realistic.

Run: python phys_metrics.py [AURORA DENSE MUSES]
"""
import os, sys, json, glob, numpy as np, cv2
from scipy.stats import wasserstein_distance

ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
POOLS = os.path.join(ROOT, "pools")
FEATS = os.path.join(ROOT, "phys_feats")
os.makedirs(FEATS, exist_ok=True)
WEATHERS = ["clear", "fog", "rain", "snow"]
FEAT_NAMES = ["dcp", "contrast", "airlight", "sat", "transmit", "hf", "colorful", "bright", "spots"]
H = 288  # resize height for cross-dataset consistency


def features(bgr):
    h, w = bgr.shape[:2]
    scale = H / h
    bgr = cv2.resize(bgr, (int(w * scale), H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # dark channel (patch min of channel min)
    mn = bgr.min(2)
    dcp = cv2.erode(mn, np.ones((15, 15), np.uint8))
    dcp_mean = float(dcp.mean())
    airlight = float(np.quantile(gray, 0.999))
    contrast = float(gray.std())
    hsv = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = float(hsv[:, :, 1].mean() / 255.0)
    transmit = float(np.clip(1 - 0.95 * dcp / max(airlight, 1e-3), 0, 1).mean())
    hf = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    R, G, B = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]
    rg = R - G; yb = 0.5 * (R + G) - B
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    bright = float(gray.mean())
    # white top-hat -> small bright blobs (lens particles)
    th = cv2.morphologyEx((gray * 255).astype(np.uint8), cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    spots = float((th > 30).mean())
    return [dcp_mean, contrast, airlight, sat, transmit, hf, colorful, bright, spots]


def pool_files(ds, w):
    return sorted(glob.glob(os.path.join(POOLS, ds, w, "*.png")) +
                  glob.glob(os.path.join(POOLS, ds, w, "*.jpg")))


def get_feats(ds, w, force=False):
    cache = os.path.join(FEATS, f"{ds}__{w}.npy")
    if os.path.exists(cache) and not force:
        return np.load(cache)
    files = pool_files(ds, w)
    if not files:
        return None
    F = []
    for i, f in enumerate(files):
        im = cv2.imread(f)
        if im is None:
            continue
        F.append(features(im))
    F = np.array(F, np.float32)
    np.save(cache, F)
    print(f"  [phys] {ds}/{w}: {F.shape}", flush=True)
    return F


def main():
    want = sys.argv[1:] or ["AURORA", "DENSE", "MUSES"]
    feats = {}
    for ds in want:
        for w in WEATHERS:
            if pool_files(ds, w):
                f = get_feats(ds, w)
                if f is not None and len(f) >= 10:
                    feats[(ds, w)] = f

    # per-pool mean/std table
    table = {}
    for (ds, w), F in feats.items():
        table[f"{ds}/{w}"] = {FEAT_NAMES[i]: [round(float(F[:, i].mean()), 4), round(float(F[:, i].std()), 4)]
                              for i in range(len(FEAT_NAMES))}

    # per-feature pooled std (for normalizing W1 across features)
    allF = np.concatenate(list(feats.values()), 0)
    fstd = allF.std(0) + 1e-8

    reals = [k for k in feats if k[0] in ("DENSE", "MUSES")]
    out = {"pools": {f"{d}/{w}": int(v.shape[0]) for (d, w), v in feats.items()},
           "feature_table": table, "w1_matrix": {}, "verdict": {}}

    for (rd, rw) in reals:
        col = {}
        for aw in WEATHERS:
            if ("AURORA", aw) not in feats:
                continue
            # normalized aggregate W1 across features
            per = {FEAT_NAMES[i]: round(wasserstein_distance(feats[("AURORA", aw)][:, i],
                                                             feats[(rd, rw)][:, i]) / fstd[i], 3)
                   for i in range(len(FEAT_NAMES))}
            agg = round(float(np.mean(list(per.values()))), 3)
            col[aw] = {"agg_W1": agg, "per_feat": per}
        tgt = f"{rd}/{rw}"
        out["w1_matrix"][tgt] = col
        best = min(col, key=lambda k: col[k]["agg_W1"])
        out["verdict"][tgt] = {"closest_AURORA_weather": best, "matched_is_closest": best == rw,
                               "agg_W1_matched": col.get(rw, {}).get("agg_W1"),
                               "agg_W1_clear": col.get("clear", {}).get("agg_W1")}

    with open(os.path.join(ROOT, "phys_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["verdict"], indent=2))
    print("wrote phys_results.json")


if __name__ == "__main__":
    main()
