#!/usr/bin/env python
"""
Post-processing of snow_rescue_raw.json:
  - per-bin + far-range-pooled (>=30m) dropout-rate curves, fog (control) vs snow (test)
  - spurious count-per-frame, fog (control) vs snow (test)
  - two-proportion z-tests for the far-range dropout steps (light->mod->heavy)
  - two-sample z-tests (per-frame mean/std) for spurious count-per-frame steps
  - figure: snow_rescue_curves.png/.pdf
  - snow_rescue_summary.json (verdict numbers)
No dataset access -- pure post-processing of snow_rescue_raw.json.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(OUTDIR, "snow_rescue_raw.json")))
C = d["cells"]
SEVS = ["light", "moderate", "heavy"]
LABELS = C["fog|light"]["range_bin_labels"]
FAR_IDX = list(range(3, 7))  # 30-40, 40-50, 50-60, >60


def is_monotone(vals, increasing=True):
    if any(v is None for v in vals):
        return None
    if increasing:
        return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p2 - p1) / se if se > 0 else float("nan")


def two_sample_z(m1, s1, n1, m2, s2, n2):
    se = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    return (m2 - m1) / se if se > 0 else float("nan")


summary = {}

# ---------------------------------------------------------------- (2) far-range dropout rate
print("=== Far-range (>=30m) paired dropout rate (S_c-binned) ===")
far_dropout = {}
for w in ("fog", "snow"):
    far_dropout[w] = {}
    for s in SEVS:
        clean = sum(C[f"{w}|{s}"]["clean_count_per_bin"][i] for i in FAR_IDX)
        drop = sum(C[f"{w}|{s}"]["drop_count_per_bin"][i] for i in FAR_IDX)
        far_dropout[w][s] = {"clean_n": clean, "drop_n": drop, "rate": drop / clean}
    vals = [far_dropout[w][s]["rate"] for s in SEVS]
    mono = is_monotone(vals, True)
    print(f"  {w:5s}: {[round(v,4) for v in vals]}  monotone_increasing={mono}")
    zs = {}
    for a, b in [("light", "moderate"), ("moderate", "heavy"), ("light", "heavy")]:
        z = two_prop_z(far_dropout[w][a]["drop_n"], far_dropout[w][a]["clean_n"],
                        far_dropout[w][b]["drop_n"], far_dropout[w][b]["clean_n"])
        zs[f"{a}_vs_{b}"] = float(z)
    print(f"    z-scores (two-proportion): {zs}")
    far_dropout[w]["monotonic"] = mono
    far_dropout[w]["z_scores"] = zs
summary["far_range_dropout_rate_ge30m"] = far_dropout

# per-bin curve monotonicity (full 7-bin breakdown)
print("\n=== Per-bin paired dropout rate monotonicity (all 7 bins) ===")
per_bin_mono = {}
for w in ("fog", "snow"):
    per_bin_mono[w] = {}
    for b, lab in enumerate(LABELS):
        vals = [C[f"{w}|{s}"]["dropout_rate_per_bin_paired"][b] for s in SEVS]
        mono = is_monotone(vals, True)
        per_bin_mono[w][lab] = {"values": vals, "monotonic": mono}
        print(f"  {w:5s} bin {lab:6s}: {[round(v,4) if v is not None else None for v in vals]}  mono={mono}")
summary["per_bin_dropout_rate_paired"] = per_bin_mono

# ---------------------------------------------------------------- (3) spurious count-per-frame
print("\n=== Spurious count-per-frame ===")
spur_summary = {}
for w in ("fog", "snow"):
    spur_summary[w] = {}
    means, stds, ns = [], [], []
    for s in SEVS:
        sp = C[f"{w}|{s}"]["spurious"]
        spur_summary[w][s] = {
            "mean_per_frame": sp["count_per_frame_mean"],
            "std_per_frame": sp["count_per_frame_std"],
            "n_frames": C[f"{w}|{s}"]["n_frames"],
            "mean_range_m": sp["mean_range_m"],
        }
        means.append(sp["count_per_frame_mean"])
        stds.append(sp["count_per_frame_std"])
        ns.append(C[f"{w}|{s}"]["n_frames"])
    mono = is_monotone(means, True)
    print(f"  {w:5s} mean_per_frame: {[round(m,1) for m in means]}  monotone_increasing={mono}")
    zs = {}
    idx = {s: i for i, s in enumerate(SEVS)}
    for a, b in [("light", "moderate"), ("moderate", "heavy"), ("light", "heavy")]:
        z = two_sample_z(means[idx[a]], stds[idx[a]], ns[idx[a]], means[idx[b]], stds[idx[b]], ns[idx[b]])
        zs[f"{a}_vs_{b}"] = float(z)
    print(f"    z-scores (two-sample mean): {zs}")
    mean_range_vals = [spur_summary[w][s]["mean_range_m"] for s in SEVS]
    print(f"    mean_range_m (spurious pts): {[round(v,2) for v in mean_range_vals]}  monotone={is_monotone(mean_range_vals, True)}/{is_monotone(mean_range_vals, False)}")
    spur_summary[w]["monotonic_count_per_frame"] = mono
    spur_summary[w]["z_scores_count_per_frame"] = zs
summary["spurious_count_per_frame"] = spur_summary

# ---------------------------------------------------------------- (5) perturbed error quantiles (negative result)
print("\n=== Perturbed error quantiles (p50), positive-control check ===")
pert_summary = {}
for w in ("fog", "snow"):
    p50s = [C[f"{w}|{s}"]["perturbed_error_quantiles_m"]["p50"] for s in SEVS]
    mono_inc = is_monotone(p50s, True)
    mono_dec = is_monotone(p50s, False)
    pert_summary[w] = {"p50_light_mod_heavy": p50s, "monotone_increasing": mono_inc, "monotone_decreasing": mono_dec}
    print(f"  {w:5s} p50: {[round(v,3) for v in p50s]}  mono_inc={mono_inc} mono_dec={mono_dec}")
summary["perturbed_error_p50"] = pert_summary
summary["perturbed_error_verdict"] = (
    "FAILS positive control: fog p50 is non-monotonic (light=18.25, moderate=31.57, heavy=21.51), "
    "same failure mode as range_rmse_pert_m in the original Deliverable-1 table. Not a valid "
    "severity probe -- discarded, not used for the snow verdict."
)

# ---------------------------------------------------------------- (1) literal item-1 retention (corroborating only)
print("\n=== Item-1 literal retention (independently binned by S_w), far bins ===")
item1_summary = {}
for w in ("fog", "snow"):
    item1_summary[w] = {}
    for b in FAR_IDX:
        lab = LABELS[b]
        vals = [C[f"{w}|{s}"]["retention_per_bin_literal_item1"][b] for s in SEVS]
        mono_dec = is_monotone(vals, False)
        item1_summary[w][lab] = {"values": vals, "monotone_decreasing": mono_dec}
        print(f"  {w:5s} bin {lab:6s}: {[round(v,4) if v is not None else None for v in vals]}  mono_dec={mono_dec}")
summary["item1_literal_retention_far_bins"] = item1_summary
summary["item1_verdict"] = (
    "Corroborates item 2 for snow (all 4 far bins monotone decreasing, consistent with rising "
    "dropout). NOT usable as fog's positive control on its own: heavy-fog far bins collapse to "
    "0 valid weather points (near-total attenuation), so the ratio is uninformative/degenerate "
    "there -- item 2 (paired dropout, denominator = clean count, always well-populated) is the "
    "primary metric; item 1 is reported as a corroborating cross-check only."
)

with open(os.path.join(OUTDIR, "snow_rescue_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nWrote snow_rescue_summary.json")

# ================================================================== FIGURE
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
x = np.arange(len(LABELS))
colors = {"light": "tab:green", "moderate": "tab:orange", "heavy": "tab:red"}

# top row: dropout-rate-vs-range-bin curve, fog (control) | snow (test)
for ax, w, title in zip(axes[0], ("fog", "snow"), ("Fog (positive control)", "Snow (test)")):
    for s in SEVS:
        vals = C[f"{w}|{s}"]["dropout_rate_per_bin_paired"]
        ax.plot(x, vals, "o-", color=colors[s], label=s)
    ax.axvspan(2.5, 6.5, color="grey", alpha=0.12, label="far range (>=30m)" if w == "snow" else None)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_xlabel("range bin (m), binned by true/clean range $S_c$")
    ax.set_ylabel("dropout rate (paired, $M_{drop}/M_c$)")
    ax.set_title(f"{title}: dropout rate vs. range")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

# bottom-left: far-range pooled dropout rate bars
ax = axes[1][0]
width = 0.35
xg = np.arange(len(SEVS))
fog_vals = [far_dropout["fog"][s]["rate"] for s in SEVS]
snow_vals = [far_dropout["snow"][s]["rate"] for s in SEVS]
ax.bar(xg - width / 2, fog_vals, width, color="tab:blue", label="fog")
ax.bar(xg + width / 2, snow_vals, width, color="tab:cyan", label="snow")
ax.set_xticks(xg)
ax.set_xticklabels(SEVS)
ax.set_ylabel("far-range (>=30m) pooled dropout rate")
ax.set_title("Far-range pooled dropout rate: fog vs snow")
ax.legend(fontsize=8)
for i, v in enumerate(snow_vals):
    ax.text(xg[i] + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
for i, v in enumerate(fog_vals):
    ax.text(xg[i] - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

# bottom-right: spurious count-per-frame bars
ax = axes[1][1]
fog_sp = [spur_summary["fog"][s]["mean_per_frame"] for s in SEVS]
snow_sp = [spur_summary["snow"][s]["mean_per_frame"] for s in SEVS]
fog_sp_err = [spur_summary["fog"][s]["std_per_frame"] / np.sqrt(spur_summary["fog"][s]["n_frames"]) for s in SEVS]
snow_sp_err = [spur_summary["snow"][s]["std_per_frame"] / np.sqrt(spur_summary["snow"][s]["n_frames"]) for s in SEVS]
ax.bar(xg - width / 2, fog_sp, width, yerr=fog_sp_err, capsize=3, color="tab:blue", label="fog")
ax.bar(xg + width / 2, snow_sp, width, yerr=snow_sp_err, capsize=3, color="tab:cyan", label="snow")
ax.set_xticks(xg)
ax.set_xticklabels(SEVS)
ax.set_ylabel("spurious ($M_{spur}$) points / frame (mean $\\pm$ SEM)")
ax.set_title("Spurious-return count per frame: fog vs snow")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "snow_rescue_curves.png"), dpi=150)
plt.savefig(os.path.join(OUTDIR, "snow_rescue_curves.pdf"))
print("Wrote snow_rescue_curves.png/.pdf")
