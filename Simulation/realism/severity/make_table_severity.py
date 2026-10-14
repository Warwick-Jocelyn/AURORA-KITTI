#!/usr/bin/env python3
"""
Build the booktabs LaTeX table (WSC cosine, |shift|, real |shift|, FID) for the
severity-matched AURORA vs real-weather comparison.
Run: python make_table_severity.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "shift_results_severity.json")) as fh:
    SHIFT = json.load(fh)["results"]
with open(os.path.join(HERE, "fid_results_severity.json")) as fh:
    FID = json.load(fh)["fid"]

ROWS = [
    ("Fog", "MUSES", "MUSES/fog"), ("Fog", "DENSE", "DENSE/fog"),
    ("Rain", "MUSES", "MUSES/rain"), ("Rain", "DENSE", "DENSE/rain"),
    ("Snow", "MUSES", "MUSES/snow"),
]
VARIANTS = [("AURORA-light", "light"), ("AURORA-lightmod", "light+mod"), ("AURORA-all", "all (orig.)")]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Severity-matched AURORA-KITTI vs real adverse weather (MUSES / DENSE). "
             r"WSC cosine and shift magnitude use the same normalization frame as the original "
             r"all-severity study (App.~X); \emph{light} and \emph{light+mod} restrict the AURORA "
             r"pool to the light (resp.\ light+moderate) severity label before recomputing the "
             r"clear$\to$weather physical shift. Real |shift| is fixed per weather/dataset (unaffected "
             r"by AURORA subsetting).}")
lines.append(r"\label{tab:severity_realism}")
lines.append(r"\begin{tabular}{llrrrrr}")
lines.append(r"\toprule")
lines.append(r"Weather & Real ref. & AURORA subset & WSC cosine $\uparrow$ & AURORA $|$shift$|$ & Real $|$shift$|$ & FID $\downarrow$ \\")
lines.append(r"\midrule")

for weather, ref, key in ROWS:
    row = SHIFT[key]
    real_mag = row["real_shift_mag"]
    fid_row = FID.get(key, {})
    for vi, (vkey, vlabel) in enumerate(VARIANTS):
        v = row["variants"][vkey]
        w_cell = weather if vi == 0 else ""
        r_cell = ref if vi == 0 else ""
        fid_val = fid_row.get(vkey)
        fid_str = f"{fid_val:.1f}" if fid_val is not None else "--"
        lines.append(f"{w_cell} & {r_cell} & {vlabel} & {v['cosine']:+.3f} & {v['shift_mag']:.2f} & {real_mag:.2f} & {fid_str} \\\\")
    if (weather, ref) != (ROWS[-1][0], ROWS[-1][1]):
        lines.append(r"\addlinespace")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

out = "\n".join(lines)
with open(os.path.join(HERE, "severity_table.tex"), "w") as fh:
    fh.write(out + "\n")
print(out)
print("\nwrote", os.path.join(HERE, "severity_table.tex"))
