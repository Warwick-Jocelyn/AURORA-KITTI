#!/usr/bin/env python3
"""
Build per-(dataset, weather) RGB image pools as symlink folders under Realism/pools/.
Sources:
  AURORA  : DMD3C val_selection_cropped_{VAL,TEST}/image  (weather from *_small.csv)
  DENSE   : DMD3C val_selection_cropped_DENSE/image        (weather from dense_small.csv)
  MUSES   : organized separately by prep_muses.py once the download finishes.
Weather folders are lowercase: clear / fog / rain / snow.
Run: python prep_pools.py
"""
import os, csv, glob

DMD3C = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
OUT = "/path/to/aurora-root/PhD_Project_2025/Realism/pools"
WEATHERS = ["clear", "fog", "rain", "snow"]


def aurora_file(name):
    # csv name  drive_image_CAM_FRAME  ->  file drive_image_FRAME_image_CAM.png
    drive, rest = name.split("_image_")
    cam, frame = rest.split("_")
    return f"{drive}_image_{frame}_image_{cam}.png"


def link(src, dstdir, dstname):
    os.makedirs(dstdir, exist_ok=True)
    dst = os.path.join(dstdir, dstname)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_aurora():
    counts = {w: 0 for w in WEATHERS}
    for split, csvn in [("val_selection_cropped_VAL", "val_small.csv"),
                        ("val_selection_cropped_TEST", "test_small.csv")]:
        root = os.path.join(DMD3C, split)
        imgdir = os.path.join(root, "image")
        with open(os.path.join(root, csvn)) as fh:
            for r in csv.DictReader(fh):
                w = r["weather"].strip().lower()
                if w not in WEATHERS:
                    continue
                f = aurora_file(r["name"].strip())
                src = os.path.join(imgdir, f)
                if not os.path.exists(src):
                    continue
                tod = (r.get("Time_of_Day") or "?").strip().lower()
                link(src, os.path.join(OUT, "AURORA", w), f"{split[-4:]}_{tod}_{f}")
                counts[w] += 1
    return counts


def build_dense():
    counts = {w: 0 for w in WEATHERS}
    root = os.path.join(DMD3C, "val_selection_cropped_DENSE")
    imgdir = os.path.join(root, "image")
    with open(os.path.join(root, "dense_small.csv")) as fh:
        for r in csv.DictReader(fh):
            w = r["weather"].strip().lower()
            if w not in WEATHERS:
                continue
            src = os.path.join(imgdir, r["name"].strip() + ".png")
            if not os.path.exists(src):
                continue
            link(src, os.path.join(OUT, "DENSE", w), r["name"].strip() + ".png")
            counts[w] += 1
    return counts


if __name__ == "__main__":
    a = build_aurora()
    d = build_dense()
    print("AURORA pools:", a, "total", sum(a.values()))
    print("DENSE  pools:", d, "total", sum(d.values()))
    for ds in ("AURORA", "DENSE"):
        for w in WEATHERS:
            n = len(glob.glob(os.path.join(OUT, ds, w, "*.png")))
            if n:
                print(f"  {ds}/{w}: {n}")
