#!/usr/bin/env python3
"""Assemble a self-contained realism.html (figures embedded as base64) from the
JSON results + figures. Run after make_figs.py."""
import os, json, base64

ROOT = "/path/to/aurora-root/PhD_Project_2025/Realism"
FIG = os.path.join(ROOT, "figures")
fid = json.load(open(os.path.join(ROOT, "fid_results.json")))
sh = json.load(open(os.path.join(ROOT, "shift_results.json")))
ph = json.load(open(os.path.join(ROOT, "phys_results.json")))


def img(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:640px;width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0">'


WE = ["clear", "fog", "rain", "snow"]


def fid_table():
    targets = list(fid["matrix"].keys())
    rows = "".join(
        "<tr><td class='m'>AURORA-%s</td>%s</tr>" % (
            aw, "".join(
                (lambda v, best: f"<td class='{('best' if best else '')}'>{v:.0f}</td>")(
                    fid["matrix"][t][aw]["FID"],
                    fid["matrix"][t][aw]["FID"] == min(c["FID"] for c in fid["matrix"][t].values()))
                if aw in fid["matrix"][t] else "<td>—</td>"
                for t in targets))
        for aw in WE)
    th = "".join(f"<th>{t}</th>" for t in targets)
    return f"<table><thead><tr><th>FID ↓</th>{th}</tr></thead><tbody>{rows}</tbody></table>"


def wsc_table():
    r = ""
    for t, v in sh["per_weather"].items():
        cls = "best" if v["cosine"] >= 0.55 else ("warn" if v["cosine"] < 0.3 else "")
        r += f"<tr><td class='m'>{t}</td><td class='{cls}'>{v['cosine']:+.3f}</td><td>{v['sign_agreement']:.2f}</td><td>{v['n_real']}</td></tr>"
    return f"<table><thead><tr><th>real target</th><th>WSC cosine ↑</th><th>sign-agree</th><th>n</th></tr></thead><tbody>{r}</tbody></table>"


CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:26px;max-width:1050px;color:#1a1a1a;background:#fafafa}
h1{font-size:23px}h2{margin-top:26px;border-bottom:2px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #e3e3e3;padding:5px 10px;text-align:right}th{background:#2c3e50;color:#fff}
td.m{text-align:left;font-weight:600}td.best{background:#d6f5d6;font-weight:700;color:#11651b}
td.warn{background:#fde2e2;color:#9b2335}.note{color:#555;font-size:13px;line-height:1.5;max-width:900px}
.grid{display:flex;flex-wrap:wrap;gap:14px}.k{background:#eef4fb;border-left:4px solid #2c3e50;padding:8px 14px;margin:8px 0;font-size:13px}"""

html = f"""<!doctype html><meta charset='utf-8'><title>AURORA-KITTI weather realism</title><style>{CSS}</style>
<h1>Quantitative validation of AURORA-KITTI weather realism</h1>
<p class='note'>Answers the reviewer's "dataset realism mostly qualitative" concern. Augments <b>Table 1</b>
(qualitative dataset comparison) and <b>§3.2 / Fig. 4</b> (visual examples). Real references: <b>MUSES</b>
(clean automotive RGB) and <b>DENSE</b> (gated/LUT camera).</p>
<div class='k'><b>Headline:</b> synthetic <b>fog</b> aligns with real fog on both datasets
(shift-cosine 0.77–0.78, lowest FID/KID of any synthetic weather); <b>snow</b> matches real snow in
direction (100% feature-sign agreement); <b>rain</b> agrees with DENSE (0.61), while MUSES rain is
physically ≈ its own clear (mild rain).</div>

<h2>1. Weather-Shift Consistency (domain-invariant, our headline metric)</h2>
<p class='note'>Cosine between the <i>clear→weather</i> physical-feature shift of our data vs. the real
dataset. Static camera/content offsets cancel. ↑ better.</p>
{wsc_table()}
<div class='grid'>{img('fig_wsc_confusion_MUSES.png')}{img('fig_wsc_confusion_DENSE.png')}</div>

<h2>2. FID cross-weather matrix</h2>
<p class='note'>For each real target (columns), distance from each AURORA weather (rows). Green = column-best.
Matched fog is best for MUSES-fog; the MUSES rain/snow columns favour clear because real MUSES rain/snow
are mild (see report §2.3–2.4).</p>
{fid_table()}
<div class='grid'>{img('fig_fid_heatmap_MUSES.png')}{img('fig_fid_heatmap_DENSE.png')}</div>

<h2>3. Physical clear→weather shift (AURORA vs real)</h2>
<p class='note'>Same-direction bars ⇒ our synthetic weather changes the image physics like real weather does.</p>
<div class='grid'>{img('fig_shift_bars_MUSES.png')}{img('fig_shift_bars_DENSE.png')}</div>

<p class='note'>Full write-up: <code>report.md</code>. LaTeX table: <code>realism_table.tex</code>.
Reproduce: <code>prep_pools.py, prep_muses.py, compute_fid.py, phys_metrics.py, analyze_shift.py, make_figs.py</code>.</p>
"""
open(os.path.join(ROOT, "realism.html"), "w").write(html)
print("wrote realism.html (%d KB)" % (len(html) // 1024))
