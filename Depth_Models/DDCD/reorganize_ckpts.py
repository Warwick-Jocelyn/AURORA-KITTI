"""
After eval_all_ckpts.py finishes, this script:
  1) reads ckpt_eval_results.jsonl,
  2) ranks the unique checkpoints by RMSE on DENSE val,
  3) moves every non-protected checkpoint into a single folder
     (checkpoints/_unified/) with a name encoded as
        rank{R}_rmse{RMSE:.0f}_{slug}.pth
     (lower RMSE = better rank),
  4) leaves the two protected (paper) checkpoints in place,
  5) writes checkpoints/REGISTRY.md with the leaderboard table.

The unified folder is the new home for every retrained checkpoint.
"""
import json
import os
import re
import shutil
import sys

ROOT = "/path/to/aurora-root/PhD_Project_2025/DMD3C"
DMD3C = os.path.join(ROOT, "BP-Net", "DMD3C")
RESULTS = os.path.join(DMD3C, "ckpt_eval_results.jsonl")
UNIFIED_DIR = os.path.join(DMD3C, "checkpoints", "_unified")
REGISTRY = os.path.join(DMD3C, "checkpoints", "REGISTRY.md")

# Original-paper checkpoints — keep names + locations untouched.
PROTECTED = {
    os.path.join(DMD3C, "checkpoints", "pretrained_mixed_singleview_256.pth"),
    os.path.join(DMD3C, "checkpoint", "BP_KITTI_noisy_disp",
                 "dmd3c_distillation_depth_anything_v2.pth"),
}


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def main():
    if not os.path.isfile(RESULTS):
        sys.exit(f"results file not found: {RESULTS}")
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # split protected vs renamable
    valid = [r for r in rows if "RMSE" in r and "error" not in r]
    valid.sort(key=lambda r: r["RMSE"])

    os.makedirs(UNIFIED_DIR, exist_ok=True)

    # Metrics that should be reported as value * 1000 with 2 decimals
    # (depth metrics: m -> mm; inverse-depth: 1/m -> 1/km).
    SCALED_x1000 = {"RMSE", "MAE", "iRMSE", "iMAE"}

    def fmt_metric(name, raw):
        if not isinstance(raw, (int, float)):
            return str(raw) if raw is not None else "—"
        if name in SCALED_x1000:
            return f"{raw * 1000.0:.2f}"
        return f"{raw:.4f}"

    # build new layout — protected stays, others move into _unified/
    leaderboard = []
    rank_n = 0
    for r in valid:
        src = r["ckpt"]
        protected = src in PROTECTED
        rmse_mm = r["RMSE"] * 1000.0  # mm
        if protected:
            new_path = src
        else:
            rank_n += 1
            slug = slugify(r["label"])
            fname = f"rank{rank_n:02d}_rmse{rmse_mm:07.2f}mm_{slug}.pth"
            new_path = os.path.join(UNIFIED_DIR, fname)
            if os.path.abspath(src) != os.path.abspath(new_path):
                if os.path.exists(new_path):
                    os.remove(new_path)
                shutil.move(src, new_path)

        leaderboard.append((rmse_mm, r, new_path, protected))

    # error rows (failed eval) — list separately
    error_rows = [r for r in rows if "error" in r]

    # ---------------- write REGISTRY.md ----------------
    cols = ["RMSE", "MAE", "iRMSE", "iMAE", "REL",
            "D^1", "D^2", "D^3", "D102", "D105", "D110",
            "PSNR", "SSIM", "MS-SSIM"]
    col_headers = []
    for c in cols:
        if c in SCALED_x1000:
            unit = "mm" if c in {"RMSE", "MAE"} else "1/km"
            col_headers.append(f"{c} ({unit})")
        else:
            col_headers.append(c)

    lines = [
        "# Checkpoint registry (DMD3C)",
        "",
        "Leaderboard ranked by **RMSE** on `datas/kitti/val_selection_cropped_DENSE`",
        "(1600 images, 14-metric MetricALL).",
        "",
        "Reported with the KITTI-depth-completion convention:",
        "- `RMSE` and `MAE` shown in **mm** (= raw value × 1000), 2 decimals.",
        "- `iRMSE` and `iMAE` shown in **1/km** (= raw value × 1000), 2 decimals.",
        "- All other metrics shown raw with 4 decimals.",
        "",
        "Two checkpoints are kept under their original names because they come from",
        "the upstream papers / pretraining and external code expects those exact paths:",
        "- `checkpoints/pretrained_mixed_singleview_256.pth` — PMP backbone init",
        "- `checkpoint/BP_KITTI_noisy_disp/dmd3c_distillation_depth_anything_v2.pth` — DMD3C distill init",
        "",
        "All other retrained checkpoints have been moved to `checkpoints/_unified/` and",
        "renamed `rank{NN}_rmse{RMSE_mm}mm_{label}.pth` so that filesystem order = leaderboard order.",
        "",
        "## Leaderboard",
        "",
        "| Rank | New name | Original source | " + " | ".join(col_headers) + " |",
        "|------|----------|-----------------|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for i, (rmse_mm, r, new_path, protected) in enumerate(leaderboard, 1):
        rel_new = os.path.relpath(new_path, DMD3C)
        rel_old = os.path.relpath(r["ckpt"], DMD3C)
        if protected:
            rel_new = f"**[paper]** {rel_new}"
        vals = [fmt_metric(c, r.get(c)) for c in cols]
        lines.append(f"| {i} | `{rel_new}` | `{rel_old}` | " + " | ".join(vals) + " |")

    if error_rows:
        lines += ["", "## Failed evaluations", ""]
        for r in error_rows:
            lines.append(f"- `{os.path.relpath(r['ckpt'], DMD3C)}` — {r.get('error')}")

    lines += [
        "",
        "## How to read RMSE",
        "",
        "RMSE is computed in metres on pixels with `gt > 1e-4`.",
        "Lower is better. iRMSE/iMAE are computed in inverse-depth space (1/m).",
        "D^k = fraction of valid pixels with `max(gt/pred, pred/gt) < 1.25^k`.",
        "PSNR / SSIM / MS-SSIM use depth maps as grayscale images, max_depth=80m.",
        "",
        "Generated by `reorganize_ckpts.py` from `ckpt_eval_results.jsonl`.",
    ]
    with open(REGISTRY, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REGISTRY}")
    print(f"\nLeaderboard (RMSE in mm):")
    for i, (rmse_mm, r, new_path, protected) in enumerate(leaderboard, 1):
        tag = "PAPER" if protected else "MOVED"
        print(f"  {i:2d}. [{tag}] RMSE={rmse_mm:8.2f}mm  {os.path.relpath(new_path, DMD3C)}")


if __name__ == "__main__":
    main()
