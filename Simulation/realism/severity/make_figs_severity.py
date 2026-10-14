#!/usr/bin/env python3
"""
Figure: shift-magnitude convergence + WSC cosine, by AURORA severity subset,
vs the real (MUSES / DENSE) reference, per weather.

Two panels:
  (a) clear->weather shift MAGNITUDE: AURORA-light / light+moderate / all vs real |shift|
  (b) WSC cosine: AURORA-light / light+moderate / all vs real target

Saves figures/fig_severity_convergence.{png,pdf}
Run: python make_figs_severity.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(HERE, "shift_results_severity.json")) as fh:
    D = json.load(fh)["results"]

# order: for each weather, we plot both MUSES and DENSE(if available) as separate groups
GROUPS = [
    ("fog", "MUSES/fog"), ("fog", "DENSE/fog"),
    ("rain", "MUSES/rain"), ("rain", "DENSE/rain"),
    ("snow", "MUSES/snow"),
]
VARIANTS = ["AURORA-light", "AURORA-lightmod", "AURORA-all"]
VLABELS = ["light", "light+mod", "all (orig.)"]
COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# --- panel (a): shift magnitude ---
ax = axes[0]
x = np.arange(len(GROUPS))
width = 0.22
for vi, v in enumerate(VARIANTS):
    ys = [D[key]["variants"][v]["shift_mag"] for w, key in GROUPS]
    ax.bar(x + (vi - 1) * width, ys, width, label=VLABELS[vi], color=COLORS[vi])
real_ys = [D[key]["real_shift_mag"] for w, key in GROUPS]
ax.scatter(x, real_ys, marker="D", s=70, color="black", zorder=5, label="real |shift|")
for xi, ry in zip(x, real_ys):
    ax.plot([xi - 1.6 * width, xi + 1.6 * width], [ry, ry], color="black", lw=1.2, ls="--", zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([key.replace("/", "\n") for w, key in GROUPS], fontsize=9)
ax.set_ylabel("clear -> weather shift magnitude (normalized units)")
ax.set_title("(a) Shift magnitude: AURORA severity subset vs real")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# --- panel (b): WSC cosine ---
ax = axes[1]
for vi, v in enumerate(VARIANTS):
    ys = [D[key]["variants"][v]["cosine"] for w, key in GROUPS]
    ax.bar(x + (vi - 1) * width, ys, width, label=VLABELS[vi], color=COLORS[vi])
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([key.replace("/", "\n") for w, key in GROUPS], fontsize=9)
ax.set_ylabel("Weather-Shift Consistency (cosine)")
ax.set_title("(b) WSC cosine: AURORA severity subset vs real")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Severity-matched AURORA vs real weather (MUSES/DENSE): magnitude converges toward real\n"
             "as severity narrows to light; MUSES-rain stays near-degenerate (property of the reference)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIG, "fig_severity_convergence.png"), dpi=180)
fig.savefig(os.path.join(FIG, "fig_severity_convergence.pdf"))
print("wrote", os.path.join(FIG, "fig_severity_convergence.png"))
