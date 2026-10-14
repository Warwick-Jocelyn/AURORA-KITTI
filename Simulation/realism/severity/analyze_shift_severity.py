#!/usr/bin/env python3
"""
Weather-Shift Consistency (WSC), recomputed for AURORA severity subsets
(light-only, light+moderate) alongside the original AURORA-all, vs the SAME
real references (MUSES primary, DENSE where available).

Methodologically important: the per-feature normalization (mu, sd) used to make
the 9 physical features comparable in scale is FIXED to the frame fit on the
*original* Realism study's 12 pools (AURORA all-severity x4 weathers, DENSE x3,
MUSES x4). We do NOT refit mu/sd when adding the severity-subset arrays (which
are strict subsets of AURORA-all's fog/rain/snow pools) -- refitting would
mean each variant is scored in a slightly different, self-referential unit
system, defeating direct comparison. Fixing the frame lets the "AURORA-all"
row reproduce the original report's numbers bit-for-bit (sanity check printed
below) and lets |shift| be read on one consistent scale across variants.

clear is shared unchanged across all three AURORA variants (severity is always
"none" for clear).

Run: python analyze_shift_severity.py
"""
import os, json, glob, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
ORIG_FEATS = os.path.join(ORIG_ROOT, "phys_feats")
NEW_FEATS = os.path.join(HERE, "phys_feats")
NAMES = ["dcp", "contrast", "airlight", "sat", "transmit", "hf", "colorful", "bright", "spots"]
WEATHERS = ["fog", "rain", "snow"]
AURORA_VARIANTS = {
    "AURORA-light": "AURORA-light",
    "AURORA-lightmod": "AURORA-lightmod",
    "AURORA-all": "AURORA",  # original, unfiltered
}
REAL_DATASETS = ["MUSES", "DENSE"]

# Reference numbers from the original (all-severity) report, for a printed sanity check.
ORIG_REPORT_COSINE = {
    ("DENSE", "fog"): 0.78, ("DENSE", "rain"): 0.61,
    ("MUSES", "fog"): 0.77, ("MUSES", "rain"): -0.32, ("MUSES", "snow"): 0.57,
}
ORIG_REPORT_REAL_SHIFT = {
    ("DENSE", "fog"): 1.59, ("DENSE", "rain"): 2.01,
    ("MUSES", "fog"): 1.73, ("MUSES", "rain"): 0.26, ("MUSES", "snow"): 0.86,
}


def load_orig():
    d = {}
    for f in glob.glob(os.path.join(ORIG_FEATS, "*.npy")):
        tag = os.path.basename(f)[:-4]
        ds, w = tag.split("__")
        d[(ds, w)] = np.load(f)
    return d


def load_new():
    d = {}
    for f in glob.glob(os.path.join(NEW_FEATS, "*.npy")):
        tag = os.path.basename(f)[:-4]
        ds, w = tag.split("__")
        d[(ds, w)] = np.load(f)
    return d


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    orig = load_orig()      # AURORA (all), DENSE, MUSES -- 12 arrays, exactly as original study
    new = load_new()        # AURORA-light, AURORA-lightmod -- 6 arrays

    # --- FIX the normalization frame to the ORIGINAL 12-array fit (see docstring) ---
    allF_orig = np.concatenate(list(orig.values()), 0)
    mu, sd = allF_orig.mean(0), allF_orig.std(0) + 1e-8

    normed = {}
    for k, v in orig.items():
        normed[k] = (v - mu) / sd
    for k, v in new.items():
        normed[k] = (v - mu) / sd

    def mean_feat(ds, w):
        return normed[(ds, w)].mean(0)

    clear_ref = mean_feat("AURORA", "clear")  # shared clear reference for all AURORA variants

    def aurora_shift(variant_key, w):
        ds = AURORA_VARIANTS[variant_key]
        if (ds, w) not in normed:
            return None
        return mean_feat(ds, w) - clear_ref

    def real_shift(rd, w):
        if (rd, w) not in normed or (rd, "clear") not in normed:
            return None
        return mean_feat(rd, w) - mean_feat(rd, "clear")

    out = {"normalization": "fixed to original 12-pool fit (see docstring)",
           "n_per_pool": {f"{ds}/{w}": int(v.shape[0]) for (ds, w), v in {**orig, **new}.items()},
           "sanity_check_vs_original_report": {},
           "results": {}}

    print("=== Sanity check: AURORA-all reproduces the original report numbers? ===")
    for (rd, w), orig_cos in ORIG_REPORT_COSINE.items():
        sa = aurora_shift("AURORA-all", w)
        sr = real_shift(rd, w)
        if sa is None or sr is None:
            continue
        c = round(cos(sa, sr), 3)
        m = round(float(np.linalg.norm(sr)), 3)
        ok_c = abs(c - orig_cos) < 0.02
        ok_m = abs(m - ORIG_REPORT_REAL_SHIFT[(rd, w)]) < 0.02
        out["sanity_check_vs_original_report"][f"{rd}/{w}"] = {
            "recomputed_cosine": c, "report_cosine": orig_cos, "cosine_match": ok_c,
            "recomputed_real_shift_mag": m, "report_real_shift_mag": ORIG_REPORT_REAL_SHIFT[(rd, w)],
            "shift_match": ok_m}
        print(f"  {rd}/{w}: cosine {c} (report {orig_cos}) {'OK' if ok_c else 'MISMATCH'}; "
              f"real|shift| {m} (report {ORIG_REPORT_REAL_SHIFT[(rd, w)]}) {'OK' if ok_m else 'MISMATCH'}")

    print("\n=== Severity-subset WSC ===")
    for rd in REAL_DATASETS:
        for w in WEATHERS:
            sr = real_shift(rd, w)
            if sr is None:
                continue
            real_mag = float(np.linalg.norm(sr))
            row = {"real_shift_mag": round(real_mag, 3), "n_real": int(orig[(rd, w)].shape[0]),
                   "variants": {}}
            for vk in ["AURORA-light", "AURORA-lightmod", "AURORA-all"]:
                sa = aurora_shift(vk, w)
                if sa is None:
                    continue
                c = round(cos(sa, sr), 3)
                mag = round(float(np.linalg.norm(sa)), 3)
                sign_agree = round(float(np.mean(np.sign(sa) == np.sign(sr))), 3)
                n = int(new.get((AURORA_VARIANTS[vk], w), orig.get((AURORA_VARIANTS[vk], w))).shape[0])
                row["variants"][vk] = {"cosine": c, "shift_mag": mag, "sign_agreement": sign_agree, "n": n}
                print(f"  {rd}/{w:5s} {vk:16s}: cosine={c:+.3f}  |shift|={mag:.3f}  "
                      f"(real |shift|={real_mag:.3f}, n={n})")
            out["results"][f"{rd}/{w}"] = row

    with open(os.path.join(HERE, "shift_results_severity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote", os.path.join(HERE, "shift_results_severity.json"))


if __name__ == "__main__":
    main()
