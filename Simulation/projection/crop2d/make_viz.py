#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable visualization for clean vs weather sparse LiDAR (and, later, method
comparison: predicted dense depth vs GT in LiDAR-point regions).
Outputs live under <TEST>/viz/ (self-contained; safe to clear/regenerate).
Planned extension: add a predicted-depth row + error-on-noise-region heatmap so
different methods can be compared visually in the dropped/spurious regions.

Per sampled frame it renders a stacked panel:
  1) weather RGB
  2) CLEAN sparse depth (color)         -- velodyne_raw_clean/
  3) WEATHER sparse depth (color)       -- velodyne_raw/
  4) diff over RGB: dropped=red (clean-only), spurious=blue (weather-only), kept=green
and an index.html with per-frame stats (dropped / spurious / far-point counts).
"""
import os, csv, numpy as np, cv2

BASE = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST"
CSVP = f"{BASE}/test_small.csv"
CLEAN, WEATHER, IMG = f"{BASE}/velodyne_raw_clean", f"{BASE}/velodyne_raw", f"{BASE}/image"
OUT = f"{BASE}/viz"; os.makedirs(OUT, exist_ok=True)
PER_GROUP = 2  # frames per (weather,severity)


def colorize(depth_m, maxd=80.0):
    d = np.clip(depth_m / maxd, 0, 1)
    cm = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[depth_m <= 0] = (0, 0, 0)
    return cv2.dilate(cm, np.ones((3, 3), np.uint8))  # fatten sparse points for visibility


def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    rows = list(csv.DictReader(open(CSVP)))
    groups, picked = {}, []
    for r in rows:
        key = (r["weather"].strip(), r["severity"].strip())
        groups.setdefault(key, [])
        if len(groups[key]) < PER_GROUP:
            groups[key].append(r); picked.append(r)

    cards = []
    for r in picked:
        name = r["name"]; drive, rest = name.split("_image_"); cam, frame = rest.split("_")
        vn = f"{drive}_velodyne_raw_{frame}_image_{cam}.png"
        inm = f"{drive}_image_{frame}_image_{cam}.png"
        C = cv2.imread(f"{CLEAN}/{vn}", cv2.IMREAD_UNCHANGED)
        W = cv2.imread(f"{WEATHER}/{vn}", cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(f"{IMG}/{inm}", cv2.IMREAD_COLOR)
        if C is None or W is None:
            continue
        if rgb is None:
            rgb = np.zeros((*C.shape, 3), np.uint8)
        if rgb.shape[:2] != C.shape[:2]:
            rgb = cv2.resize(rgb, (C.shape[1], C.shape[0]))
        Cm, Wm = C.astype(np.float64) / 255.0, W.astype(np.float64) / 255.0
        cmask, wmask = C > 0, W > 0
        dropped = cmask & ~wmask; spurious = wmask & ~cmask; kept = cmask & wmask
        diff = rgb.copy()
        diff[cv2.dilate(kept.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 180, 0)
        diff[cv2.dilate(dropped.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 0, 255)
        diff[cv2.dilate(spurious.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 0, 0)

        tag = f"{r['weather'].strip()}/{r['severity'].strip()}/{r['Time_of_Day'].strip()}"
        panel = np.vstack([
            label(rgb, f"{tag}  RGB (weather)"),
            label(colorize(Cm), f"CLEAN sparse  pts={int(cmask.sum())} far>30m={int((Cm>30).sum())}"),
            label(colorize(Wm), f"WEATHER sparse  pts={int(wmask.sum())} far>30m={int((Wm>30).sum())}"),
            label(diff, f"diff  dropped(red)={int(dropped.sum())}  spurious(blue)={int(spurious.sum())}"),
        ])
        fn = f"{tag.replace('/','_')}__{frame}_{cam}.png"
        cv2.imwrite(f"{OUT}/{fn}", panel)
        cards.append((tag, fn, int(dropped.sum()), int(spurious.sum())))

    cards.sort()
    html = ["<html><head><meta charset='utf-8'><title>clean vs weather sparse</title>",
            "<style>body{background:#111;color:#ddd;font-family:sans-serif}img{width:100%;border:1px solid #333}"
            ".c{margin:18px 0}</style></head><body><h2>Clean vs Weather sparse LiDAR (TEST)</h2>",
            "<p>Rows per panel: weather RGB / CLEAN sparse / WEATHER sparse / diff "
            "(<span style='color:#f44'>dropped</span>, <span style='color:#48f'>spurious</span>, "
            "<span style='color:#4c4'>kept</span>).</p>"]
    for tag, fn, dr, sp in cards:
        html.append(f"<div class='c'><b>{tag}</b> &nbsp; dropped={dr} spurious={sp}<br><img src='{fn}'></div>")
    html.append("</body></html>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    print(f"wrote {len(cards)} panels + index.html to {OUT}")


if __name__ == "__main__":
    main()
