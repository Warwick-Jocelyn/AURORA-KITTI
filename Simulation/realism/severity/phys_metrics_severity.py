#!/usr/bin/env python3
"""
Recompute physical weather features (same 9-dim descriptor as the original
Realism/phys_metrics.py, imported directly for byte-identical feature math) on
AURORA severity subsets: light-only and light+moderate-only, for fog/rain/snow.

clear is NOT recomputed here: severity is always "none" for clear (verified),
so AURORA-clear is identical across all severity variants and we just reuse the
original cached Realism/phys_feats/AURORA__clear.npy.

Caches to ./phys_feats/AURORA-light__<w>.npy and AURORA-lightmod__<w>.npy
Run: python phys_metrics_severity.py
"""
import os, sys, json, glob, numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
POOLS = os.path.join(ORIG_ROOT, "pools")
FEATS = os.path.join(HERE, "phys_feats")
os.makedirs(FEATS, exist_ok=True)

sys.path.insert(0, ORIG_ROOT)
from phys_metrics import features, FEAT_NAMES  # noqa: E402  (reuse exact feature math)

WEATHERS = ["fog", "rain", "snow"]
VARIANTS = {
    "AURORA-light": {"light"},
    "AURORA-lightmod": {"light", "moderate"},
}


def main():
    with open(os.path.join(HERE, "severity_map.json")) as fh:
        sev = json.load(fh)

    summary = {}
    for variant, sevset in VARIANTS.items():
        for w in WEATHERS:
            cache = os.path.join(FEATS, f"{variant}__{w}.npy")
            files = sorted(f for f, s in sev[w].items() if s in sevset)
            paths = [os.path.join(POOLS, "AURORA", w, f) for f in files]
            paths = [p for p in paths if os.path.exists(p)]
            if os.path.exists(cache):
                F = np.load(cache)
            else:
                F = []
                for p in paths:
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    F.append(features(im))
                F = np.array(F, np.float32)
                np.save(cache, F)
            summary[f"{variant}/{w}"] = int(F.shape[0])
            print(f"[phys-severity] {variant}/{w}: n={F.shape[0]} (from {len(paths)} matched files)")

    with open(os.path.join(HERE, "severity_pool_counts.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", os.path.join(HERE, "severity_pool_counts.json"))


if __name__ == "__main__":
    main()
