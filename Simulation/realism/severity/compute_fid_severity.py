#!/usr/bin/env python3
"""
FID(AURORA-light / AURORA-lightmod, MUSES-<w>) vs the original FID(AURORA-all, MUSES-<w>),
for fog/rain/snow. Reuses the ALREADY-CACHED Inception activations for MUSES/DENSE from the
original study (Realism/feats/*.npy) -- only the AURORA severity-subset activations are new.

CPU-ONLY BY CONSTRUCTION: CUDA_VISIBLE_DEVICES is cleared before importing torch, so this
process cannot see or touch the GPU (a 26h training job is running there). Do not remove
that line.

Run: python compute_fid_severity.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard guarantee: no GPU visibility, must stay first

import sys, json, glob, time, numpy as np, torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
ORIG_FEATS = os.path.join(ORIG_ROOT, "feats")
POOLS = os.path.join(ORIG_ROOT, "pools")
NEW_FEATS = os.path.join(HERE, "feats")
os.makedirs(NEW_FEATS, exist_ok=True)
assert not torch.cuda.is_available(), "refusing to run: CUDA became visible, aborting to protect the GPU training job"
DEV = "cpu"
WEATHERS = ["fog", "rain", "snow"]
VARIANTS = {"AURORA-light": {"light"}, "AURORA-lightmod": {"light", "moderate"}}


def fid_from_feats(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False); s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    with open(os.path.join(HERE, "severity_map.json")) as fh:
        sev = json.load(fh)

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEV).eval()

    new_feats = {}
    for variant, sevset in VARIANTS.items():
        for w in WEATHERS:
            cache = os.path.join(NEW_FEATS, f"{variant}__{w}.npy")
            if os.path.exists(cache):
                new_feats[(variant, w)] = np.load(cache)
                continue
            files = sorted(f for f, s in sev[w].items() if s in sevset)
            paths = [os.path.join(POOLS, "AURORA", w, f) for f in files]
            paths = [p for p in paths if os.path.exists(p)]
            t0 = time.time()
            act = get_activations(paths, model, batch_size=50, dims=2048, device=DEV, num_workers=4)
            np.save(cache, act)
            new_feats[(variant, w)] = act
            print(f"[fid-severity] {variant}/{w}: {act.shape[0]} imgs in {time.time()-t0:.0f}s", flush=True)

    # load cached real + original AURORA-all activations
    orig_feats = {}
    for f in glob.glob(os.path.join(ORIG_FEATS, "*.npy")):
        tag = os.path.basename(f)[:-4]
        ds, w = tag.split("__")
        orig_feats[(ds, w)] = np.load(f)

    out = {"fid": {}}
    reals = [k for k in orig_feats if k[0] in ("MUSES", "DENSE")]
    for (rd, rw) in reals:
        if rw not in WEATHERS:
            continue
        row = {"n_real": int(orig_feats[(rd, rw)].shape[0])}
        row["AURORA-all"] = round(fid_from_feats(orig_feats[("AURORA", rw)], orig_feats[(rd, rw)]), 1)
        for variant in VARIANTS:
            if (variant, rw) in new_feats:
                row[variant] = round(fid_from_feats(new_feats[(variant, rw)], orig_feats[(rd, rw)]), 1)
        out["fid"][f"{rd}/{rw}"] = row
        print(f"{rd}/{rw}: {row}")

    with open(os.path.join(HERE, "fid_results_severity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", os.path.join(HERE, "fid_results_severity.json"))


if __name__ == "__main__":
    main()
