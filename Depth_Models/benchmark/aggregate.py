#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all per-method JSON results (json/<method>__<SET>.json) into:
  - results.md           (markdown tables)
  - results.html         (self-contained styled website; serve over HTTP)
  - tables.tex           (LaTeX booktabs tables for the rebuttal)
Re-runnable as results accumulate. Unknown methods are still shown (appended).

method_meta.json (same dir) supplies Params (M) and Time (ms) columns.
"""
import os, json, glob, math

RZB = os.path.dirname(os.path.abspath(__file__))
JDIR = os.environ.get("JDIR", os.path.join(RZB, "json"))
# OUTSUF lets a second (appendix) run write tables_<suf>.tex etc. without clobbering
# the main outputs, e.g. OUTSUF=_full352 JDIR=.../json_full352 python aggregate.py
OUTSUF = os.environ.get("OUTSUF", "")

# ---- Load method_meta.json (Params + Time) ----------------------------------
_META_PATH = os.path.join(RZB, "method_meta.json")
try:
    _META_RAW = json.load(open(_META_PATH))
    _META_RAW.pop("_note", None)
except Exception:
    _META_RAW = {}

def meta_params(tag):
    """Return Params (M) as formatted string, or '--'."""
    m = _META_RAW.get(tag, {})
    v = m.get("params_M")
    if v is None:
        return "--"
    if v == 0:
        return "0"
    return f"{v:.1f}"

def meta_time(tag):
    """Return Time (ms) as formatted string, or '--'."""
    m = _META_RAW.get(tag, {})
    v = m.get("time_ms")
    if v is None:
        return "--"
    if v >= 10000:
        return f"{v/1000:.0f}s"  # e.g. Marigold-DC ~34s
    return f"{v:.0f}"

def meta_modality(tag):
    """Return modality string from method_meta, or '--'."""
    return _META_RAW.get(tag, {}).get("modality", "--")

_MOD_TEX = {"LiDAR": "L", "RGB": "RGB", "RGB+LiDAR": "RGB+L"}
_MOD_BADGE_STYLE = {
    "LiDAR":     "background:#e67e22;color:#fff",
    "RGB":       "background:#2980b9;color:#fff",
    "RGB+LiDAR": "background:#27ae60;color:#fff",
}

def mod_tex(tag):
    """Return abbreviated modality string for TeX (L / RGB / RGB+L)."""
    return _MOD_TEX.get(meta_modality(tag), "?")

def mod_html_badge(tag):
    """Return a small inline HTML badge for the modality."""
    m = meta_modality(tag)
    style = _MOD_BADGE_STYLE.get(m, "background:#aaa;color:#fff")
    return (f'<span style="padding:1px 5px;border-radius:3px;font-size:11px;'
            f'font-weight:700;white-space:nowrap;{style}">{m}</span>')

# canonical display order & metadata: key tag -> (display, category, family)
# category: zeroshot | trained | ours
REG = [
    ("IPBasic-fast",            "IP-Basic (Fast)",            "zeroshot", "classical"),
    ("IPBasic-multiscale",      "IP-Basic (Multiscale)",      "zeroshot", "classical"),
    ("S2D-Sup-ZeroShot",        "Sparse2Dense (Sup.)",        "zeroshot", "s2d"),
    ("S2D-SelfSup-ZeroShot",    "Sparse2Dense (Self-sup.)",   "zeroshot", "s2d"),
    ("NLSPN-ZeroShot",          "NLSPN",                      "zeroshot", "spn"),
    ("CSPNpp-ZeroShot",         "CSPN++",                     "zeroshot", "spn"),
    ("CompletionFormer-ZeroShot","CompletionFormer",          "zeroshot", "transformer"),
    ("BPNet",                   "BP-Net",                     "zeroshot", "bp"),
    ("DMD3C-Pretrained",        "DMD3C (pretrained)",         "zeroshot", "dmd3c"),
    ("DMD3C-distilled",         "DMD3C (distilled)",          "zeroshot", "dmd3c"),
    ("Marigold-DC",             "Marigold-DC",                "zeroshot", "diffusion"),
    ("DepthAnythingV3",         "Depth-Anything-V3",          "zeroshot", "mono"),
    ("UniK3D",                  "UniK3D",                     "zeroshot", "mono"),
    ("OMNI-DC",                 "OMNI-DC",                    "zeroshot", "omnidc"),
    ("S2D-SelfSup-AURORA",      "Sparse2Dense (Self-sup., AURORA-ft)", "trained", "s2d"),
    ("Sparse2Dense-Sup-AURORA", "Sparse2Dense (Sup., AURORA-ft)",      "trained", "s2d"),
    ("CompletionFormer-AURORA", "CompletionFormer (AURORA-ft)",        "trained", "transformer"),
    ("DDCD",                    "DDCD (ours)",                "ours",     "ours"),
    # Appendix retrain: teacher supervises the full 352 frame incl. top-96 (no crop
    # of the DA pseudo-label). Auto-appears once its json is dropped into a json dir.
    ("DDCD-h352",               "DDCD (ours, top-supervised)","ours",     "ours"),
]
DISP = {k: v for k, v, _, _ in REG}
CAT = {k: c for k, _, c, _ in REG}
ORDER = [k for k, _, _, _ in REG]

STD = [("RMSE", "RMSE↓ (mm)", -1), ("MAE", "MAE↓ (mm)", -1),
       ("iRMSE", "iRMSE↓ (1/km)", -1), ("iMAE", "iMAE↓ (1/km)", -1)]
# Proposed denoising metrics (SR and NRR removed per rebuttal decision 2026-07-01:
# SR is monotone with ERR and adds no ranking signal; NRR is a diagnostic ratio
# that a uniformly-inaccurate method also scores ~1, so it misleads in a results
# table). Kept: ERR (error-reduction), DCE (dropout-completion error), CPP (clean-point).
PROP = [("ERR", "ERR↑", +1), ("DCE", "DCE↓ (mm)", -1), ("CPP", "CPP (mm)", -1)]


def load_all(jdir=None):
    data = {}  # set -> tag -> json
    for f in glob.glob(os.path.join(jdir or JDIR, "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        data.setdefault(j["set"], {})[j["method"]] = j
    return data


def tags_for(set_, data):
    present = list(data.get(set_, {}).keys())
    ordered = [t for t in ORDER if t in present] + [t for t in present if t not in ORDER]
    return ordered


def val(j, metric, group="ALL"):
    g = j.get("groups", {}).get(group)
    if not g or metric not in g:
        return None
    v = g[metric]
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def best_set(data, set_, tags, metric, direction):
    if direction is None:  # diagnostic metric (e.g. NRR) — never highlight a "best"
        return set()
    vals = [(t, val(data[set_][t], metric)) for t in tags]
    vals = [(t, v) for t, v in vals if v is not None]
    if not vals:
        return set()
    if direction > 0:
        m = max(v for _, v in vals)
    elif direction < 0:
        m = min(v for _, v in vals)
    else:  # NRR closest to 1
        m = min(vals, key=lambda x: abs(x[1] - 1.0))[1]
        return {t for t, v in vals if abs(v - 1.0) == abs(m - 1.0)}
    return {t for t, v in vals if v == m}


def fmt(v, metric):
    if v is None:
        return "—"
    if metric in ("ERR",):
        return f"{v:.3f}"
    return f"{v:.1f}"

# Extra column headers and values for efficiency
EXTRA_COLS = [("Params (M)", "params"), ("Time (ms)", "time")]

def extra_vals(tag):
    """Return (params_str, time_str) for a method tag."""
    return meta_params(tag), meta_time(tag)


# ---------------- Markdown ----------------
def md_table(data, set_):
    tags = tags_for(set_, data)
    if not tags:
        return f"### {set_}\n\n_(no results yet)_\n"
    cols = STD + (PROP if set_ == "TEST" else [])
    bests = {m: best_set(data, set_, tags, m, d) for m, _, d in cols}
    extra_hdrs = " | ".join(lbl for lbl, _ in EXTRA_COLS)
    hdr = "| Method | Cat | Modality | " + " | ".join(lbl for _, lbl, _ in cols) + " | " + extra_hdrs + " |"
    sep = "|" + "---|" * (len(cols) + 3 + len(EXTRA_COLS))
    lines = [f"### {set_} (overall, group=ALL)", "", hdr, sep]
    for t in tags:
        j = data[set_][t]
        row = [DISP.get(t, t), CAT.get(t, "?"), meta_modality(t)]
        for m, _, _ in cols:
            v = val(j, m)
            s = fmt(v, m)
            if t in bests[m] and v is not None:
                s = f"**{s}**"
            row.append(s)
        p_str, tm_str = extra_vals(t)
        row += [p_str, tm_str]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------- LaTeX ----------------
def tex_table(data, set_):
    tags = tags_for(set_, data)
    if not tags:
        return ""
    cols = STD + (PROP if set_ == "TEST" else [])
    bests = {m: best_set(data, set_, tags, m, d) for m, _, d in cols}
    n_extra = len(EXTRA_COLS)
    # colspec: l=Method, c=Modality, r*n=metrics+extra
    colspec = "lc" + "r" * (len(cols) + n_extra)
    metric_hdrs = [lbl.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$").replace("→1", r"$\to1$").replace("_", r"\_") for _, lbl, _ in cols]
    extra_hdrs = [lbl for lbl, _ in EXTRA_COLS]
    head = " & ".join(["Method", "Mod."] + metric_hdrs + extra_hdrs)
    n_cols_total = 1 + 1 + len(cols) + n_extra  # Method + Modality + metrics + extra
    out = [r"\begin{table*}[t]\centering",
           r"\caption{%s split: standard%s metrics (overall). Best per column in \textbf{bold}. "
           r"Mod.: input modality (L=LiDAR-only, RGB=image-only, RGB+L=RGB+LiDAR). "
           r"Params (M): total trainable parameters. Time (ms): single-frame, batch=1, GPU-only, fp32, "
           r"RTX~4090, median of 50 runs (CUDA Events); IP-Basic is non-learned (CPU). "
           r"DDCD uses checkpoint \texttt{rank01\_rmse2802.79mm\_dmd3d\_bp\_kitti\_best\_ema\_new\_loss.pth}.%s}" %
           (set_, " and proposed denoising" if set_ == "TEST" else "",
            (" \\emph{Protocol:} full-$352$ frame (no top crop), applied uniformly to all methods."
             if OUTSUF else
             " \\emph{Protocol:} DENSE uses the BP-Net selval crop (bottom $256$ rows) applied uniformly to all methods; TEST is full $352$ (its top $96$ rows carry no valid GT)." )),
           r"\label{tab:bench_%s%s}" % (set_.lower(), OUTSUF),
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{%s}" % colspec, r"\toprule", head + r" \\", r"\midrule"]
    last_cat = None
    for t in tags:
        cat = CAT.get(t, "?")
        if cat != last_cat:
            tag = {"zeroshot": "Zero-shot (pretrained)", "trained": "Trained on AURORA-KITTI", "ours": "Ours"}.get(cat, cat)
            out.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (n_cols_total, tag))
            last_cat = cat
        j = data[set_][t]
        cells = [DISP.get(t, t).replace("_", r"\_"), mod_tex(t)]
        for m, _, _ in cols:
            v = val(j, m)
            s = fmt(v, m)
            if t in bests[m] and v is not None:
                s = r"\textbf{%s}" % s
            cells.append(s)
        p_str, tm_str = extra_vals(t)
        cells += [p_str, tm_str]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(out)


# ---------------- HTML ----------------
def html_table(data, set_, title=None, subtitle=None):
    tags = tags_for(set_, data)
    if not tags:
        return f"<h2>{title or set_}</h2><p><em>no results yet</em></p>"
    cols = STD + (PROP if set_ == "TEST" else [])
    bests = {m: best_set(data, set_, tags, m, d) for m, _, d in cols}
    th = "<th>Modality</th>" + "".join(f"<th>{lbl}</th>" for _, lbl, _ in cols)
    th += "".join(f"<th>{lbl}</th>" for lbl, _ in EXTRA_COLS)
    n_all_cols = 1 + 1 + len(cols) + len(EXTRA_COLS)  # Method + Modality + metrics + extra
    rows = []
    last_cat = None
    for t in tags:
        cat = CAT.get(t, "?")
        if cat != last_cat:
            tag = {"zeroshot": "Zero-shot (pretrained)", "trained": "Trained on AURORA-KITTI", "ours": "Ours"}.get(cat, cat)
            rows.append(f'<tr class="cat"><td colspan="{n_all_cols}">{tag}</td></tr>')
            last_cat = cat
        j = data[set_][t]
        tds = [f'<td style="text-align:center">{mod_html_badge(t)}</td>']
        for m, _, _ in cols:
            v = val(j, m)
            s = fmt(v, m)
            cls = "best" if (t in bests[m] and v is not None) else ""
            tds.append(f'<td class="{cls}">{s}</td>')
        p_str, tm_str = extra_vals(t)
        tds.append(f'<td class="eff">{p_str}</td>')
        tds.append(f'<td class="eff">{tm_str}</td>')
        cls_row = "ours" if cat == "ours" else ""
        rows.append(f'<tr class="{cls_row}"><td class="m">{DISP.get(t,t)}</td>' + "".join(tds) + "</tr>")
    head = title or f"{set_} <small>(overall, group=ALL)</small>"
    sub = f"<p class='note' style='margin-top:-4px'>{subtitle}</p>" if subtitle else ""
    return f"<h2>{head}</h2>{sub}<table><thead><tr><th>Method</th>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def html_perweather(data, set_):
    """Per-weather standard metrics (RMSE/MAE) for context."""
    tags = tags_for(set_, data)
    weathers = ["Clear", "Fog", "Rain", "Snow"]
    blocks = []
    for metric, lbl, d in [("RMSE", "RMSE↓ (mm)", -1)]:
        th = "".join(f"<th>{w}</th>" for w in weathers)
        rows = []
        for t in tags:
            j = data[set_][t]
            tds = []
            for w in weathers:
                v = val(j, metric, w)
                tds.append(f"<td>{fmt(v, metric)}</td>")
            rows.append(f'<tr><td class="m">{DISP.get(t,t)}</td>' + "".join(tds) + "</tr>")
        blocks.append(f"<h3>{set_} — per-weather {lbl}</h3><table><thead><tr><th>Method</th>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
    return "".join(blocks)


def build():
    data = load_all()
    sets = [s for s in ("TEST", "DENSE") if s in data]
    # markdown
    md = ["# AURORA-KITTI benchmark — all methods × all metrics\n",
          "TEST = AURORA-KITTI (paired clean LiDAR → standard + proposed). DENSE = real SeeingThroughFog (standard only).\n",
          "Proposed denoising metrics: ERR↑ (error-reduction), SR↑ (suppression), DCE↓ (dropout-completion err), NRR→1, CPP.\n"]
    for s in sets:
        md.append(md_table(data, s))
    open(os.path.join(RZB, f"results{OUTSUF}.md"), "w").write("\n".join(md))
    # tex. For the full-352 appendix (OUTSUF set) emit only the DENSE table —
    # the TEST split is identical to the main table (TEST is always full 352).
    tex_sets = [s for s in sets if s == "DENSE"] if OUTSUF else sets
    tex = ["% auto-generated by aggregate.py — \\input this into the rebuttal",
           "% requires booktabs, graphicx (resizebox)"]
    for s in tex_sets:
        tex.append(tex_table(data, s))
        tex.append("")
    open(os.path.join(RZB, f"tables{OUTSUF}.tex"), "w").write("\n".join(tex))
    # html
    css = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
h1{font-size:22px}h2{margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}small{color:#888;font-weight:normal}
table{border-collapse:collapse;margin:10px 0 24px;font-size:13px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #e3e3e3;padding:5px 9px;text-align:right}th{background:#2c3e50;color:#fff;position:sticky;top:0}
td.m{text-align:left;font-weight:600;white-space:nowrap}
td.best{background:#d6f5d6;font-weight:700;color:#11651b}
td.eff{color:#555;background:#fafafa}
tr.cat td{background:#eef2f7;font-style:italic;font-weight:700;text-align:left;color:#2c3e50}
tr.ours td.m{color:#b30059}tr.ours{background:#fff5fa}
.note{color:#666;font-size:13px;max-width:900px;line-height:1.5}
.defbox{background:#f0f4fa;border-left:4px solid #2c3e50;padding:10px 16px;margin:12px 0;max-width:900px;font-size:13px}
.defbox h3{margin:0 0 6px;font-size:14px;color:#2c3e50}.defbox dl{margin:0}.defbox dt{font-weight:700;display:inline}
.defbox dd{display:inline;margin:0}.defbox dd::after{content:"\\A";white-space:pre}"""
    DDCD_CKPT = (
        "checkpoints/_unified/"
        "rank01_rmse2802.79mm_dmd3d_bp_kitti_best_ema_new_loss.pth"
        " (inside PhD_Project_2025/DMD3C/BP-Net/DMD3C/)"
    )
    metrics_panel = (
        "<div class='defbox'>"
        "<h3>Proposed denoising metrics — plain-language definitions</h3>"
        "<dl>"
        "<dt>ERR&uarr; (Error-Reduction Rate):</dt><dd> 1 &minus; RMSE(prediction) / RMSE(corrupted input) on pixels where LiDAR was corrupted. "
        "ERR=1 = perfect correction; ERR=0 = output as bad as corrupted input; ERR&lt;0 = amplification.</dd>"
        "<dt>DCE&darr; (Dropout Completion Error):</dt><dd> RMSE on pixels where real LiDAR was dropped by weather "
        "(fog/snow erasure); measures completion accuracy under structured, weather-induced missingness.</dd>"
        "<dt>CPP&darr; (Clean-Point Precision):</dt><dd> RMSE on pixels where the clean LiDAR point was correctly preserved; "
        "measures fine-range accuracy on unaffected returns.</dd>"
        "</dl></div>"
    )
    provenance_note = (
        "<p class='note'><b>DDCD checkpoint:</b> "
        f"<code>{DDCD_CKPT}</code> (symlinked from BP-Net/checkpoints/DDCD/result_ema.pth; "
        "confirmed via Hydra run logs 2026-05-03). "
        "<b>Time (ms) protocol:</b> single-frame, batch=1, GPU forward-only, fp32, RTX 4090, "
        "median of 50 runs after 10 warmup (CUDA Events); IP-Basic uses CPU perf_counter.</p>"
    )
    parts = [f"<!doctype html><meta charset='utf-8'><title>AURORA-KITTI benchmark</title><style>{css}</style>",
             "<h1>AURORA-KITTI — full benchmark (all methods × all metrics)</h1>",
             "<p class='note'>&#128202; <a href='vis/figures.html'><b>Figures: do the proposed metrics reveal completion + denoising?</b></a></p>",
             "<p class='note'>TEST = AURORA-KITTI (has paired clean LiDAR → standard <b>and</b> proposed denoising metrics). "
             "DENSE = real SeeingThroughFog (no clean LiDAR → standard metrics only). "
             "Proposed metrics: <b>ERR↑</b> error-reduction on corrupted returns, "
             "<b>DCE↓</b> dropout-completion error, <b>CPP↓</b> clean-point error. "
             "Green = best per column. "
             "<b>DENSE is shown under both crop protocols</b> (bottom-256 and full-352); "
             "within each table every method uses the identical protocol.</p>",
             metrics_panel,
             provenance_note]
    # TEST first (full 352, standard + proposed metrics)
    if "TEST" in data:
        parts.append(html_table(data, "TEST"))
    # DENSE under BOTH crop protocols, each internally uniform across all methods.
    data352 = load_all(os.path.join(RZB, "json_full352"))
    if "DENSE" in data:
        parts.append(html_table(
            data, "DENSE",
            title="DENSE — bottom-256 crop <small>(BP-Net selval protocol; all methods)</small>",
            subtitle="Top 96 rows cropped from GT and predictions for every method (matches the AURORA-KITTI "
                     "training resolution). This is the main-paper protocol; DDCD reproduces the paper's 2802.80&nbsp;mm.")
        )
    if "DENSE" in data352:
        parts.append(html_table(
            data352, "DENSE",
            title="DENSE — full 352, no crop <small>(uncropped; all methods)</small>",
            subtitle="Full frame evaluated for every method (top 96 rows kept). Baselines reproduce the paper's "
                     "Table&nbsp;5 exactly (UniK3D 3296.9, S2D-Sup 4903.5); the current DDCD checkpoint was trained "
                     "with the top cropped, so its top region is unsupervised — see the top-supervised retrain below.")
        )
    # per-weather RMSE context (main 256-crop DENSE + TEST)
    for s in sets:
        parts.append(html_perweather(data, s))
    open(os.path.join(RZB, f"results{OUTSUF}.html"), "w").write("\n".join(parts))
    # console summary
    for s in sets:
        print(f"[{s}] methods: {len(tags_for(s, data))}  -> " + ", ".join(tags_for(s, data)))
    print("wrote results.md, results.html, tables.tex")


if __name__ == "__main__":
    build()
