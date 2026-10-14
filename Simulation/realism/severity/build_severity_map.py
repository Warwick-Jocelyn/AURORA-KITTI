#!/usr/bin/env python3
"""
Map every pooled AURORA image (Realism/pools/AURORA/<weather>/*.png) back to its
severity label (light / moderate / heavy) from the DMD3C val_selection_cropped_{VAL,TEST}
CSVs, using the EXACT same naming convention prep_pools.py used to build the pool
symlinks, so the join is unambiguous and verifiable:

  pool filename = f"{split[-4:]}_{tod}_{aurora_file(name)}"
  aurora_file(name): "drive_image_CAM_FRAME" -> "drive_image_FRAME_image_CAM.png"

We rebuild that exact string per CSV row (not the reverse-parse direction) and use
it as the join key against the actual files present in the pool directories. This
guarantees a filename either joins exactly or is reported as unmatched (there is no
fuzzy join).

Output: severity_map.json  {"fog": {filename: severity, ...}, "rain": {...}, "snow": {...}}
        (clear is skipped: severity is always "none" for clear, i.e. no subsetting needed)
Also prints a verification summary (matched / unmatched counts per weather per split).
"""
import os, csv, json, glob

DMD3C = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
POOLS = "/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Realism/pools"
OUT = os.path.dirname(os.path.abspath(__file__))
WEATHERS = ["fog", "rain", "snow"]


def aurora_file(name):
    drive, rest = name.split("_image_")
    cam, frame = rest.split("_")
    return f"{drive}_image_{frame}_image_{cam}.png"


def main():
    severity_map = {w: {} for w in WEATHERS}
    row_counts = {w: 0 for w in WEATHERS}
    unmatched = {w: 0 for w in WEATHERS}

    for split, csvn in [("val_selection_cropped_VAL", "val_small.csv"),
                        ("val_selection_cropped_TEST", "test_small.csv")]:
        root = os.path.join(DMD3C, split)
        with open(os.path.join(root, csvn)) as fh:
            for r in csv.DictReader(fh):
                w = r["weather"].strip().lower()
                if w not in WEATHERS:
                    continue
                sev = r["severity"].strip().lower()
                f = aurora_file(r["name"].strip())
                tod = (r.get("Time_of_Day") or "?").strip().lower()
                pool_fname = f"{split[-4:]}_{tod}_{f}"
                pool_path = os.path.join(POOLS, "AURORA", w, pool_fname)
                row_counts[w] += 1
                if os.path.exists(pool_path) or os.path.islink(pool_path):
                    severity_map[w][pool_fname] = sev
                else:
                    unmatched[w] += 1

    # cross-check against actual pool directory listing
    print("Verification (CSV rows -> pool files):")
    for w in WEATHERS:
        n_pool = len(glob.glob(os.path.join(POOLS, "AURORA", w, "*.png")))
        n_mapped = len(severity_map[w])
        print(f"  {w}: csv_rows={row_counts[w]} matched={n_mapped} unmatched={unmatched[w]} "
              f"pool_files_on_disk={n_pool} {'OK' if n_mapped == n_pool else 'MISMATCH'}")
        from collections import Counter
        c = Counter(severity_map[w].values())
        print(f"     severity breakdown: {dict(c)}")

    with open(os.path.join(OUT, "severity_map.json"), "w") as fh:
        json.dump(severity_map, fh, indent=2)
    print("wrote", os.path.join(OUT, "severity_map.json"))


if __name__ == "__main__":
    main()
