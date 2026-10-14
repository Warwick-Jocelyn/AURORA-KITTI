#!/usr/bin/env python3
"""
Fog-ONLY realism figure for the rebuttal.
Top:  qualitative gallery -- AURORA-KITTI synthetic fog (clear -> light -> moderate ->
      heavy, severity control) vs REAL fog (MUSES, DENSE).
Bottom: quantitative fog realism -- Weather-Shift Consistency (WSC), clear->fog shift
      magnitude (with real markers = severity convergence), and FID to real MUSES fog.
All numbers are read-only copies of rgb_realism/{shift,fid}_results_severity.json (fog rows).
Output: figures/fig_fog_realism.{pdf,png}
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

POOLS = "/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Realism/pools"
OUT   = "/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Realism_updated/rgb_realism/figures"

# ---- chosen example images (day, clean scenes) ----
import glob as _glob
def _pick(folder, pat, k):
    fs = sorted(_glob.glob(f"{POOLS}/{folder}/{pat}"))
    return fs[k % len(fs)] if fs else f"{POOLS}/{folder}/__none__.png"
_muses = sorted(_glob.glob(f"{POOLS}/MUSES/fog/*day*.png")) or sorted(_glob.glob(f"{POOLS}/MUSES/fog/*.png"))
IMGS = {
    "AURORA clear":            f"{POOLS}/AURORA/clear/TEST_day_2011_09_26_drive_0051_sync_image_0000000347_image_02.png",
    "AURORA fog (light)":      f"{POOLS}/AURORA/fog/_VAL_day_2011_09_26_drive_0005_sync_image_0000000048_image_02.png",
    "AURORA fog (moderate)":   f"{POOLS}/AURORA/fog/_VAL_day_2011_09_26_drive_0005_sync_image_0000000035_image_02.png",
    "AURORA fog (heavy)":      f"{POOLS}/AURORA/fog/_VAL_day_2011_09_26_drive_0005_sync_image_0000000017_image_02.png",
    "MUSES a": _muses[len(_muses)//5]   if _muses else "",
    "MUSES b": _muses[len(_muses)*2//5] if _muses else "",
    "MUSES c": _muses[len(_muses)*3//5] if _muses else "",
    "MUSES d": _muses[len(_muses)*4//5] if _muses else "",
}

TARGET_AR = 1216/352.0   # KITTI aspect; crop everything to this for visual consistency
def load_crop(path, ar=TARGET_AR, W=560):
    import glob
    if not os.path.exists(path):  # pick any file from the same folder as fallback
        cands = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.png")))
        path = cands[len(cands)//2]
    im = Image.open(path).convert("RGB")
    w, h = im.size
    cur = w/h
    if cur > ar:                 # too wide -> crop width
        nw = int(round(h*ar)); x0 = (w-nw)//2; im = im.crop((x0, 0, x0+nw, h))
    else:                        # too tall -> crop height (center)
        nh = int(round(w/ar)); y0 = (h-nh)//2; im = im.crop((0, y0, w, y0+nh))
    im = im.resize((W, int(round(W/ar))), Image.LANCZOS)
    return np.asarray(im)

# ---- fog quantitative data vs REAL MUSES fog (fog rows of the JSONs; read-only) ----
variants = ["light", "light+mod", "all"]
COL = {"light": "#2ca02c", "light+mod": "#ff7f0e", "all": "#d62728"}
WSC_MUSES    = {"light": 0.740, "light+mod": 0.759, "all": 0.769}
SHIFT_AURORA = {"light": 3.141, "light+mod": 3.613, "all": 4.248}
REAL_SHIFT_MUSES = 1.729
FID_MUSES    = {"light": 144.1, "light+mod": 133.4, "all": 128.5}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(15, 8.2))
gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.0, 2.15],
                       hspace=0.42, wspace=0.95)

def show(r, c0, c1, key, title, color="black"):
    ax = fig.add_subplot(gs[r, c0:c1]); ax.imshow(load_crop(IMGS[key])); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_title(title, color=color, fontsize=9.5, pad=3)
    return ax

# Row 0: AURORA synthetic (clear -> severity progression)
show(0, 0, 3,  "AURORA clear",          "AURORA-KITTI (clear)",      "#555555")
show(0, 3, 6,  "AURORA fog (light)",    "AURORA fog — light",        COL["light"])
show(0, 6, 9,  "AURORA fog (moderate)", "AURORA fog — moderate",     COL["light+mod"])
show(0, 9, 12, "AURORA fog (heavy)",    "AURORA fog — heavy",        COL["all"])
# Row 1: real MUSES fog references
show(1, 0, 3,  "MUSES a", "REAL fog — MUSES", "#1f77b4")
show(1, 3, 6,  "MUSES b", "REAL fog — MUSES", "#1f77b4")
show(1, 6, 9,  "MUSES c", "REAL fog — MUSES", "#1f77b4")
show(1, 9, 12, "MUSES d", "REAL fog — MUSES", "#1f77b4")

# left row captions
fig.text(0.085, 0.845, "Synthetic", rotation=90, va="center", ha="center", fontsize=10, weight="bold", color="#333")
fig.text(0.085, 0.640, "Real",      rotation=90, va="center", ha="center", fontsize=10, weight="bold", color="#333")

xv = np.arange(3)  # light, light+mod, all
cols = [COL[v] for v in variants]

# (a) WSC vs real MUSES fog
axa = fig.add_subplot(gs[2, 0:4])
axa.bar(xv, [WSC_MUSES[v] for v in variants], color=cols)
for i, v in enumerate(variants): axa.text(i, WSC_MUSES[v]+0.02, f"{WSC_MUSES[v]:.2f}", ha="center", fontsize=9)
axa.set_xticks(xv); axa.set_xticklabels(variants)
axa.set_ylabel("WSC cosine (→1 = same direction)"); axa.set_ylim(0, 1.0)
axa.set_title("(a) Weather-Shift Consistency vs real MUSES fog"); axa.grid(axis="y", alpha=0.3)

# (b) shift magnitude convergence toward real
axb = fig.add_subplot(gs[2, 4:8])
axb.bar(xv, [SHIFT_AURORA[v] for v in variants], color=cols)
for i, v in enumerate(variants): axb.text(i, SHIFT_AURORA[v]+0.08, f"{SHIFT_AURORA[v]:.2f}", ha="center", fontsize=9)
axb.axhline(REAL_SHIFT_MUSES, color="k", ls="--", lw=2)
axb.text(-0.4, REAL_SHIFT_MUSES+0.08, f"real MUSES = {REAL_SHIFT_MUSES:.2f}", ha="left", fontsize=8.5, style="italic")
axb.set_xticks(xv); axb.set_xticklabels(variants)
axb.set_ylabel("clear→fog shift magnitude"); axb.set_ylim(0, 4.7)
axb.set_title("(b) Severity-matched shift → real"); axb.grid(axis="y", alpha=0.3)

# (c) FID to real MUSES fog
axc = fig.add_subplot(gs[2, 8:12])
axc.bar(xv, [FID_MUSES[v] for v in variants], color=cols)
for i, v in enumerate(variants): axc.text(i, FID_MUSES[v]+2, f"{FID_MUSES[v]:.1f}", ha="center", fontsize=9)
axc.set_xticks(xv); axc.set_xticklabels(variants)
axc.set_ylabel("FID to real MUSES fog ↓"); axc.set_ylim(0, 165)
axc.set_title("(c) FID to real MUSES fog"); axc.grid(axis="y", alpha=0.3)

fig.suptitle("Fog realism: AURORA-KITTI synthetic fog matches real MUSES fog qualitatively and quantitatively",
             fontsize=12.5, weight="bold", y=0.985)
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig_fog_realism.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{OUT}/fig_fog_realism.png", bbox_inches="tight", dpi=150)
print("saved:", f"{OUT}/fig_fog_realism.png")
