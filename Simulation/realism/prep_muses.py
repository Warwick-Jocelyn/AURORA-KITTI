#!/usr/bin/env python3
"""
Organize MUSES real adverse-weather RGB into Realism/pools/MUSES/<weather>/.
MUSES encodes condition in the path:
  <root>/muses/frame_camera/<split>/<weather>/<day|night>/<name>.png
weather in {clear, fog, rain, snow}. We pool across train+val+test and day+night.
Run: python prep_muses.py
"""
import os, glob

SRC = "/path/to/aurora-root/Dataset/MUSES/extracted/muses/frame_camera"
OUT = "/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Realism/pools/MUSES"
WEATHERS = ["clear", "fog", "rain", "snow"]


def link(src, w, name):
    d = os.path.join(OUT, w)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, name)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    counts = {w: 0 for w in WEATHERS}
    for w in WEATHERS:
        for f in glob.glob(os.path.join(SRC, "*", w, "*", "*.png")):
            parts = f.split(os.sep)
            split, tod = parts[-4], parts[-2]
            link(f, w, f"{split}_{tod}_{os.path.basename(f)}")
            counts[w] += 1
    print("MUSES pools:", counts, "total", sum(counts.values()))
    for w in WEATHERS:
        n = len(glob.glob(os.path.join(OUT, w, "*.png")))
        print(f"  MUSES/{w}: {n}")


if __name__ == "__main__":
    main()
