#!/usr/bin/env python3
"""
Quantitative realism: FID + KID between AURORA (synthetic weather) and real
adverse-weather datasets (DENSE, MUSES), per weather.

Design (content-fair): we do NOT read a single AURORA-vs-real FID as "realism"
(scene content + camera differ between KITTI/DENSE/MUSES). Instead we build a
cross-weather matrix: for each REAL target weather set we measure the distance
from every AURORA weather. If AURORA-<w> is the closest AURORA set to real-<w>
(matched diagonal is the column-min), our synthetic weather-<w> is what moves
KITTI content toward the real weather-<w> appearance -> quantitative realism.

Features: pytorch-fid InceptionV3 pool3 (2048-d), extracted once per pool and
cached to feats/<tag>.npy. FID = Frechet distance. KID = unbiased polynomial
MMD^2 (x100), averaged over 100 random subsets for a stable mean +/- std.

Run:  python compute_fid.py            # all available pools
      python compute_fid.py AURORA DENSE   # subset
"""
import os, sys, json, glob, numpy as np, torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations

ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
POOLS = os.path.join(ROOT, "pools")
FEATS = os.path.join(ROOT, "feats")
os.makedirs(FEATS, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WEATHERS = ["clear", "fog", "rain", "snow"]


def pool_files(ds, w):
    return sorted(glob.glob(os.path.join(POOLS, ds, w, "*.png")) +
                  glob.glob(os.path.join(POOLS, ds, w, "*.jpg")))


def get_feats(model, ds, w, force=False):
    tag = f"{ds}__{w}"
    cache = os.path.join(FEATS, tag + ".npy")
    if os.path.exists(cache) and not force:
        return np.load(cache)
    files = pool_files(ds, w)
    if not files:
        return None
    act = get_activations(files, model, batch_size=50, dims=2048, device=DEV, num_workers=8)
    np.save(cache, act)
    print(f"  [feats] {tag}: {act.shape[0]} imgs -> {cache}", flush=True)
    return act


def fid_from_feats(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False); s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _mmd2(x, y):
    d = x.shape[1]
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m, n = x.shape[0], y.shape[0]
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return (kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1))
            - 2 * kxy.mean())


def kid_from_feats(a, b, n_sub=100, sub=1000, seed=0):
    rng = np.random.RandomState(seed)
    m = min(sub, a.shape[0], b.shape[0])
    vals = []
    for _ in range(n_sub):
        xa = a[rng.choice(a.shape[0], m, replace=False)]
        xb = b[rng.choice(b.shape[0], m, replace=False)]
        vals.append(_mmd2(xa, xb))
    vals = np.array(vals)
    return float(vals.mean() * 100), float(vals.std() * 100)


def main():
    want = sys.argv[1:] or ["AURORA", "DENSE", "MUSES"]
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEV).eval()

    # discover available pools
    feats = {}
    for ds in want:
        for w in WEATHERS:
            if pool_files(ds, w):
                f = get_feats(model, ds, w)
                if f is not None and f.shape[0] >= 10:
                    feats[(ds, w)] = f
    print("available pools:", {f"{d}/{w}": v.shape[0] for (d, w), v in feats.items()}, flush=True)

    reals = [k for k in feats if k[0] in ("DENSE", "MUSES")]
    out = {"pools": {f"{d}/{w}": int(v.shape[0]) for (d, w), v in feats.items()},
           "matrix": {}, "delta_fid": {}, "reference": {}}

    # cross matrix: rows AURORA weather, cols each real target
    for (rd, rw) in reals:
        col = {}
        for aw in WEATHERS:
            if ("AURORA", aw) in feats:
                fid = fid_from_feats(feats[("AURORA", aw)], feats[(rd, rw)])
                kmean, kstd = kid_from_feats(feats[("AURORA", aw)], feats[(rd, rw)])
                col[aw] = {"FID": round(fid, 1), "KID_x100": round(kmean, 3), "KID_std": round(kstd, 3)}
        tgt = f"{rd}/{rw}"
        out["matrix"][tgt] = col
        # delta: how much does applying our matched weather reduce the gap vs our clear?
        if "clear" in col and rw in col:
            out["delta_fid"][tgt] = {
                "FID_clear": col["clear"]["FID"], "FID_matched": col[rw]["FID"],
                "delta": round(col["clear"]["FID"] - col[rw]["FID"], 1),
                "matched_is_best": (rw == min(col, key=lambda k: col[k]["FID"]))}

    # reference gaps to calibrate scale
    def ref(a, b, label):
        if a in feats and b in feats:
            out["reference"][label] = {"FID": round(fid_from_feats(feats[a], feats[b]), 1),
                                        "KID_x100": round(kid_from_feats(feats[a], feats[b])[0], 3)}
    ref(("DENSE", "clear"), ("DENSE", "fog"), "real_within: DENSE clear->fog")
    ref(("DENSE", "fog"), ("MUSES", "fog"), "real_cross: DENSE fog->MUSES fog")
    ref(("AURORA", "clear"), ("AURORA", "fog"), "synth_within: AURORA clear->fog")
    ref(("MUSES", "clear"), ("MUSES", "fog"), "real_within: MUSES clear->fog")

    with open(os.path.join(ROOT, "fid_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote fid_results.json")


if __name__ == "__main__":
    main()
