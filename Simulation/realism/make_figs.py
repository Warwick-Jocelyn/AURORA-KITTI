#!/usr/bin/env python3
"""
Figures for the realism study. Reads fid_results.json / phys_results.json /
shift_results.json and writes PNG+PDF into Realism/figures/.
  fig_fid_heatmap    : FID (AURORA weather rows x real-weather cols), per real dataset
  fig_wsc_confusion  : Weather-Shift cosine confusion (AURORA x real), per real dataset
  fig_shift_bars     : clear->weather physical shift, AURORA vs real (matched weather)
Run: python make_figs.py
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
WE = ["clear", "fog", "rain", "snow"]
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def fid_heatmaps():
    d = json.load(open(os.path.join(ROOT, "fid_results.json")))
    targets = list(d["matrix"].keys())
    ds_groups = {}
    for t in targets:
        ds_groups.setdefault(t.split("/")[0], []).append(t)
    for ds, ts in ds_groups.items():
        rows = WE
        cols = [t.split("/")[1] for t in ts]
        M = np.full((len(rows), len(cols)), np.nan)
        for j, t in enumerate(ts):
            for i, aw in enumerate(rows):
                if aw in d["matrix"][t]:
                    M[i, j] = d["matrix"][t][aw]["FID"]
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cols), 3.6))
        im = ax.imshow(M, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"real\n{c}" for c in cols])
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"AURORA-{r}" for r in rows])
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    best = M[:, j][~np.isnan(M[:, j])].min() == M[i, j]
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                            color="white" if M[i, j] > np.nanmean(M) else "black",
                            fontweight="bold" if best else "normal",
                            fontsize=10)
                    if best:
                        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec="red", lw=2.5))
        ax.set_title(f"FID: AURORA weather vs real {ds}\n(lower=more realistic; red=best per column)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="FID")
        save(fig, f"fig_fid_heatmap_{ds}")


def wsc_confusion():
    d = json.load(open(os.path.join(ROOT, "shift_results.json")))
    for ds, blk in d["confusion_cos"].items():
        conf = blk["matrix"]; cols = list(conf.keys()); rows = WE[1:]  # fog,rain,snow
        M = np.full((len(rows), len(cols)), np.nan)
        for j, rw in enumerate(cols):
            for i, aw in enumerate(rows):
                if aw in conf[rw]:
                    M[i, j] = conf[rw][aw]
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cols), 3.4))
        im = ax.imshow(M, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"real\n{c}" for c in cols])
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"AURORA-{r}" for r in rows])
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    best = M[:, j][~np.isnan(M[:, j])].max() == M[i, j]
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=10,
                            fontweight="bold" if best else "normal")
                    if best:
                        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec="black", lw=2.5))
        ax.set_title(f"Weather-Shift Consistency (cosine) vs real {ds}\n(higher=better; black=best per column)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="cosine")
        save(fig, f"fig_wsc_confusion_{ds}")


def shift_bars():
    d = json.load(open(os.path.join(ROOT, "shift_results.json")))
    sv = d["shift_vectors"]
    feats = ["dcp", "contrast", "airlight", "sat", "transmit", "hf", "spots"]
    reals = sorted({k.split("/")[0] for k in sv if not k.startswith("AURORA")})
    for R in reals:
        ws = [w for w in ["fog", "rain", "snow"] if f"{R}/{w}" in sv and f"AURORA/{w}" in sv]
        fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 3.2), sharey=True)
        if len(ws) == 1:
            axes = [axes]
        x = np.arange(len(feats))
        for ax, w in zip(axes, ws):
            a = [sv[f"AURORA/{w}"][f] for f in feats]
            r = [sv[f"{R}/{w}"][f] for f in feats]
            ax.bar(x - 0.2, a, 0.4, label="AURORA (synth)", color="#c44")
            ax.bar(x + 0.2, r, 0.4, label=f"{R} (real)", color="#48c")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
            ax.set_title(f"clear$\\to${w}")
        axes[0].set_ylabel("normalised feature shift")
        axes[0].legend(fontsize=9)
        fig.suptitle(f"Physical clear$\\to$weather shift: AURORA vs real {R} (same direction = realistic)")
        save(fig, f"fig_shift_bars_{R}")


if __name__ == "__main__":
    fid_heatmaps(); wsc_confusion(); shift_bars()
    print("wrote figures to", FIG)
