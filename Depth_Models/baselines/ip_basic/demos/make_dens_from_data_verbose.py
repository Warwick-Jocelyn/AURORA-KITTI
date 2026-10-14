# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 让 ip_basic 可被 import（按你的路径保持一致）
sys.path.append('/path/to/aurora-root/Code/ip_basic')
from ip_basic import depth_map_utils_original

IMG_EXTS = {".png", ".PNG"}

def log(msg):
    try:
        print(msg)
    except Exception:
        # 防止非 UTF-8 终端崩溃
        print(msg.encode('utf-8', 'ignore'))
    sys.stdout.flush()

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def list_stems(folder, exts=None):
    stems = set()
    if not os.path.isdir(folder):
        return stems
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(f)
        if exts is None or ext in exts or ext.lower() in exts:
            stems.add(name)
    return stems

def has_complete_dens(data_path, dens_path, verbose=False):
    img_stems = list_stems(data_path, IMG_EXTS)
    if not img_stems:
        if verbose:
            log("  · No input PNGs under {}".format(data_path))
        return False
    dens_png_stems = set()
    if os.path.isdir(dens_path):
        for f in os.listdir(dens_path):
            p = os.path.join(dens_path, f)
            if os.path.isfile(p) and f.lower().endswith(".png"):
                dens_png_stems.add(os.path.splitext(f)[0])
    missing = img_stems - dens_png_stems
    if missing:
        if verbose:
            log("  · Missing {} outputs in {}".format(len(missing), dens_path))
        return False
    return True

def iter_kitti_camera_data(root, verbose=False):
    if not os.path.isdir(root):
        log("❌ Root not found or not a directory: {}".format(root))
        return
    found_any = False
    for date_dir in sorted(os.listdir(root)):
        date_path = os.path.join(root, date_dir)
        if not os.path.isdir(date_path):
            continue
        for drive_dir in sorted(os.listdir(date_path)):
            if not drive_dir.endswith("_sync"):
                continue
            drive_path = os.path.join(date_path, drive_dir)
            for cam in ["image_02", "image_03"]:
                cam_path = os.path.join(drive_path, cam)
                data_path = os.path.join(cam_path, "data")
                dens_path = os.path.join(cam_path, "dens")
                if os.path.isdir(data_path):
                    found_any = True
                    if verbose:
                        log("🔎 Found data: {}  -> dens: {}".format(data_path, dens_path))
                    yield data_path, dens_path
                else:
                    if verbose:
                        log("  · Skip (no data): {}".format(data_path))
    if not found_any:
        log("⚠️ No camera data folders found under root. Check the path and directory structure.")

def fill_once(projected_depths, fill_type="multiscale", extrapolate=False,
              blur_type="bilateral", show_process=False):
    if fill_type == "fast":
        final_depths = depth_map_utils_original.fill_in_fast(
            projected_depths,
            extrapolate=extrapolate,
            blur_type=blur_type
        )
    elif fill_type == "multiscale":
        final_depths, _ = depth_map_utils_original.fill_in_multiscale(
            projected_depths,
            extrapolate=extrapolate,
            blur_type=blur_type,
            show_process=show_process
        )
    else:
        raise ValueError("Invalid fill_type: {}".format(fill_type))
    return final_depths

def process_folder(data_path, dens_path, fill_type, extrapolate, blur_type, show_process,
                   skip_existing=True, progress_every=50, verbose=False):
    ensure_dir(dens_path)

    if skip_existing and has_complete_dens(data_path, dens_path, verbose=verbose):
        log("⏭️  Skip (already complete): {} -> {}".format(data_path, dens_path))
        return

    files = [f for f in sorted(os.listdir(data_path))
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    total = len(files)
    if total == 0:
        log("⚠️ No PNG inputs under: {}".format(data_path))
        return

    log("🚀 Start folder: {}".format(data_path))
    log("   · Total PNGs: {}".format(total))
    log("   · Output dir: {}".format(dens_path))

    ok_cnt, fail_cnt, skip_cnt = 0, 0, 0

    for idx, fname in enumerate(files, 1):
        stem, _ = os.path.splitext(fname)
        in_path = os.path.join(data_path, fname)
        out_path = os.path.join(dens_path, stem + ".png")

        if skip_existing and os.path.isfile(out_path):
            skip_cnt += 1
            if verbose and ((idx % progress_every == 0) or (idx == total)):
                log("   · Progress {}/{} | ok={}, skip={}, fail={}".format(idx, total, ok_cnt, skip_cnt, fail_cnt))
            continue

        depth_u16 = cv2.imread(in_path, cv2.IMREAD_ANYDEPTH)
        if depth_u16 is None:
            log("❌ Read failed: {}".format(in_path))
            fail_cnt += 1
            continue

        projected_depths = np.float32(depth_u16) / 256.0

        try:
            final_depths = fill_once(
                projected_depths,
                fill_type=fill_type,
                extrapolate=extrapolate,
                blur_type=blur_type,
                show_process=show_process
            )
        except Exception as e:
            log("❌ Fill failed: {}\n   {}".format(in_path, e))
            fail_cnt += 1
            continue

        pred_u16 = (np.clip(final_depths, 0, np.finfo(np.float32).max) * 256.0).astype(np.uint16)
        if not cv2.imwrite(out_path, pred_u16):
            log("❌ Write failed: {}".format(out_path))
            fail_cnt += 1
            continue

        ok_cnt += 1
        if (idx % progress_every == 0) or (idx == total):
            log("   · Progress {}/{} | ok={}, skip={}, fail={}".format(idx, total, ok_cnt, skip_cnt, fail_cnt))

    log("✅ Done folder: {} -> {} | ok={}, skip={}, fail={}".format(data_path, dens_path, ok_cnt, skip_cnt, fail_cnt))

def main():
    parser = argparse.ArgumentParser(description="Generate dens PNGs from KITTI data folders (CPU depth completion)")
    parser.add_argument("--root", type=str, required=True, help="Path to KITTI raw root folder")
    parser.add_argument("--fill-type", type=str, default="multiscale", choices=["fast", "multiscale"])
    parser.add_argument("--extrapolate", action="store_true")
    parser.add_argument("--blur-type", type=str, default="bilateral", choices=["gaussian", "bilateral"])
    parser.add_argument("--show-process", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (per camera folder)")
    parser.add_argument("--overwrite", action="store_true", help="Do not skip existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Print discovered paths and extra info")
    parser.add_argument("--progress-every", type=int, default=50, help="Print per-folder progress every N images")
    args = parser.parse_args()

    log("📂 Root: {}".format(args.root))
    tasks = list(iter_kitti_camera_data(args.root, verbose=args.verbose))
    log("📝 Discovered {} camera data folders.".format(len(tasks)))
    if not tasks:
        log("⚠️ No tasks to run. Double-check the --root and directory structure.")
        return

    log("🏁 Start processing with {} workers...".format(args.workers))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        total_tasks = len(tasks)
        for i, (data_path, dens_path) in enumerate(tasks, 1):
            log("  [{:03d}/{:03d}] Queue: {}".format(i, total_tasks, data_path))
            futures.append(
                ex.submit(
                    process_folder,
                    data_path, dens_path,
                    args.fill_type, args.extrapolate, args.blur_type, args.show_process,
                    skip_existing=(not args.overwrite),
                    progress_every=args.progress_every,
                    verbose=args.verbose
                )
            )
        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
                log("✔️  Task {}/{} finished.".format(i, total_tasks))
            except Exception as e:
                log("⚠️ Task {} failed with exception:\n   {}".format(i, e))

    log("🎉 All done! Dens PNGs are saved alongside each camera folder under 'dens/'.")

if __name__ == "__main__":
    main()
