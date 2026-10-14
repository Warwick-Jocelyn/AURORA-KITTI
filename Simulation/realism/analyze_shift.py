#!/usr/bin/env python3
"""
Weather-Shift Consistency (WSC) — our headline realism metric, invariant to the
static camera/content domain offset between KITTI, DENSE and MUSES.

Idea (difference-of-differences): a static offset (e.g. DENSE is globally darker
than KITTI) cancels if we compare the *shift* each weather induces relative to
that dataset's own clear condition. For dataset X and weather w:
    shift_X(w) = mean(phys | X, w) - mean(phys | X, clear)     (per feature, /std)
If our synthetic weather is realistic, shift_AURORA(w) should point in the SAME
direction (high cosine) and similar magnitude as shift_REAL(w). We also build a
confusion matrix cos(shift_AURORA(aw), shift_REAL(rw)); the matched diagonal
should dominate each real column.

Uses cached phys_feats/*.npy from phys_metrics.py. Run: python analyze_shift.py
"""
import os, json, glob, numpy as np

ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
FEATS = os.path.join(ROOT, "phys_feats")
NAMES = ["dcp", "contrast", "airlight", "sat", "transmit", "hf", "colorful", "bright", "spots"]
WEATHERS = ["fog", "rain", "snow"]


def load():
    d = {}
    for f in glob.glob(os.path.join(FEATS, "*.npy")):
        tag = os.path.basename(f)[:-4]
        ds, w = tag.split("__")
        d[(ds, w)] = np.load(f)
    return d


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    feats = load()
    allF = np.concatenate(list(feats.values()), 0)
    mu, sd = allF.mean(0), allF.std(0) + 1e-8
    norm = {k: (v - mu) / sd for k, v in feats.items()}  # normalized features

    def shift(ds, w):
        if (ds, w) not in norm or (ds, "clear") not in norm:
            return None
        return norm[(ds, w)].mean(0) - norm[(ds, "clear")].mean(0)

    out = {"per_weather": {}, "confusion_cos": {}, "shift_vectors": {}}
    reals = sorted({ds for (ds, w) in feats if ds in ("DENSE", "MUSES")})

    for R in reals:
        for w in WEATHERS:
            sa, sr = shift("AURORA", w), shift(R, w)
            if sa is None or sr is None:
                continue
            sign_agree = float(np.mean(np.sign(sa) == np.sign(sr)))
            out["per_weather"][f"{R}/{w}"] = {
                "cosine": round(cos(sa, sr), 3),
                "sign_agreement": round(sign_agree, 3),
                "n_real": int(feats[(R, w)].shape[0])}
        # confusion: rows AURORA weather, cols real weather
        conf = {}
        for rw in WEATHERS:
            sr = shift(R, rw)
            if sr is None:
                continue
            conf[rw] = {}
            for aw in WEATHERS:
                sa = shift("AURORA", aw)
                if sa is not None:
                    conf[rw][aw] = round(cos(sa, sr), 3)
        # matched-diagonal check
        diag_ok = {rw: (max(c, key=c.get) == rw) for rw, c in conf.items() if c}
        out["confusion_cos"][R] = {"matrix": conf, "matched_is_argmax": diag_ok}

    # store the raw shift vectors (for the report / a radar plot)
    for ds in ["AURORA"] + reals:
        for w in WEATHERS + ["clear"]:
            s = shift(ds, w) if w != "clear" else None
            if s is not None:
                out["shift_vectors"][f"{ds}/{w}"] = {NAMES[i]: round(float(s[i]), 3) for i in range(len(NAMES))}

    with open(os.path.join(ROOT, "shift_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["per_weather"], indent=2))
    print("\nConfusion (matched_is_argmax):")
    for R, v in out["confusion_cos"].items():
        print(f"  {R}: {v['matched_is_argmax']}")
    print("wrote shift_results.json")


if __name__ == "__main__":
    main()
