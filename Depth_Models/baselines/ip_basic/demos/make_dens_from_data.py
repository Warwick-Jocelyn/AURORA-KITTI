# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 按你的环境，确保能 import ip_basic
sys.path.append('/path/to/aurora-root/Code/ip_basic')
from ip_basic import depth_map_utils_original

IMG_EXTS = ('.png', '.PNG')

def log(msg):
    try:
        print(msg)
    except Exception:
        print(msg.encode('utf-8', 'ignore'))
    sys.stdout.flush()

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def list_pngs(folder):
    if not os.path.isdir(folder):
        return []
    files = [f for f in sorted(os.listdir(folder)) if f.endswith(IMG_EXTS)]
    return files

def stems(files):
    return set([os.path.splitext(f)[0] for f in files])

def parse_date_from_drive(drive_dir):
    """
    '2011_09_26_drive_0001_sync' -> '2011_09_26'
    """
    parts = drive_dir.split('_drive')
    if len(parts) >= 1:
        return parts[0]
    return None

def fill_once(projected_depths, fill_type, extrapolate, blur_type, show_process):
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

def has_complete_dens(input_dir, dens_dir, verbose=False):
    in_files = list_pngs(input_dir)
    if not in_files:
        if verbose:
            log("  · No input PNGs under {}".format(input_dir))
        return False
    in_stems = stems(in_files)

    out_files = list_pngs(dens_dir)
    out_stems = stems(out_files)
    missing = in_stems - out_stems
    if missing:
        if verbose:
            log("  · Missing {} outputs in {}".format(len(missing), dens_dir))
        return False
    return True

def iter_sparse_cam_dirs(sparse_root, splits, verbose=False):
    """
    遍历 data_depth_velodyne/<split>/<drive_sync>/proj_depth/velodyne_raw/image_0{2,3}
    产出条目：(split, drive_dir, cam, input_cam_dir)
    """
    base = sparse_root  # 预期传入的是 data_depth_velodyne 目录
    if not os.path.isdir(base):
        log("❌ Sparse root not found: {}".format(base))
        return

    found = False
    for split in splits:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            if verbose:
                log("  · Skip split (not found): {}".format(split_dir))
            continue

        for drive_dir in sorted(os.listdir(split_dir)):
            drive_path = os.path.join(split_dir, drive_dir)
            if not os.path.isdir(drive_path):
                continue

            # 仅接受 *_sync
            if not drive_dir.endswith("_sync"):
                if verbose:
                    log("  · Skip (not sync): {}".format(drive_dir))
                continue

            for cam in ("image_02", "image_03"):
                input_cam_dir = os.path.join(
                    drive_path, "proj_depth", "velodyne_raw", cam
                )
                if os.path.isdir(input_cam_dir):
                    found = True
                    if verbose:
                        log("🔎 Found sparse: {} -> {}".format(drive_dir, input_cam_dir))
                    yield split, drive_dir, cam, input_cam_dir
                else:
                    if verbose:
                        log("  · No sparse dir: {}".format(input_cam_dir))

    if not found:
        log("⚠️ No sparse camera dirs found under {}".format(base))

def process_cam_dir(raw_root, drive_dir, cam, input_cam_dir,
                    fill_type, extrapolate, blur_type, show_process,
                    skip_existing, progress_every):
    """
    对单个 drive+cam 的输入目录执行：读取稀疏深度 PNG (uint16, /256 m) → 稠密化 → 保存到 raw/<date>/<drive>/<cam>/dens/*.png
    """
    date = parse_date_from_drive(drive_dir)
    if date is None:
        log("❌ Cannot parse date from drive: {}".format(drive_dir))
        return

    dens_dir = os.path.join(raw_root, date, drive_dir, cam, "dens")
    ensure_dir(dens_dir)

    if skip_existing and has_complete_dens(input_cam_dir, dens_dir, verbose=True):
        log("⏭️  Skip (already complete): {} {} -> {}".format(drive_dir, cam, dens_dir))
        return

    files = list_pngs(input_cam_dir)
    total = len(files)
    if total == 0:
        log("⚠️ No PNG inputs under: {}".format(input_cam_dir))
        return

    log("🚀 Start: {} {} | total={} | out={}".format(drive_dir, cam, total, dens_dir))

    ok_cnt, fail_cnt, skip_cnt = 0, 0, 0
    for idx, fname in enumerate(files, 1):
        in_path = os.path.join(input_cam_dir, fname)
        out_path = os.path.join(dens_dir, fname)  # 同名输出

        if skip_existing and os.path.isfile(out_path):
            skip_cnt += 1
            if (idx % progress_every == 0) or (idx == total):
                log("   · Progress {}/{} | ok={}, skip={}, fail={}".format(idx, total, ok_cnt, skip_cnt, fail_cnt))
            continue

        depth_u16 = cv2.imread(in_path, cv2.IMREAD_ANYDEPTH)
        if depth_u16 is None:
            log("❌ Read failed: {}".format(in_path))
            fail_cnt += 1
            continue

        # uint16 → 米
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

        # 保存为 png (uint16, ×256)
        pred_u16 = (np.clip(final_depths, 0, np.finfo(np.float32).max) * 256.0).astype(np.uint16)
        if not cv2.imwrite(out_path, pred_u16):
            log("❌ Write failed: {}".format(out_path))
            fail_cnt += 1
            continue

        ok_cnt += 1
        if (idx % progress_every == 0) or (idx == total):
            log("   · Progress {}/{} | ok={}, skip={}, fail={}".format(idx, total, ok_cnt, skip_cnt, fail_cnt))

    log("✅ Done: {} {} -> {} | ok={}, skip={}, fail={}".format(drive_dir, cam, dens_dir, ok_cnt, skip_cnt, fail_cnt))

def main():
    parser = argparse.ArgumentParser(description="Densify KITTI sparse velodyne depth into raw/<date>/<drive>/<cam>/dens as PNG")
    parser.add_argument("--sparse-root", type=str, required=True,
                        help="Path to data_depth_velodyne (the directory that contains train/test/val)")
    parser.add_argument("--raw-root", type=str, required=True,
                        help="Path to KITTI raw root (e.g., .../datas/kitti/raw)")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to include (default: train,val,test)")
    parser.add_argument("--fill-type", type=str, default="multiscale", choices=["fast", "multiscale"])
    parser.add_argument("--extrapolate", action="store_true")
    parser.add_argument("--blur-type", type=str, default="bilateral", choices=["gaussian", "bilateral"])
    parser.add_argument("--show-process", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    log("📂 Sparse root: {}".format(args.sparse_root))
    log("📁 Raw root   : {}".format(args.raw_root))
    log("🧩 Splits     : {}".format(", ".join(splits)))

    entries = list(iter_sparse_cam_dirs(args.sparse_root, splits, verbose=args.verbose))
    log("📝 Discovered {} (split, drive, cam) input folders.".format(len(entries)))
    if not entries:
        log("⚠️ No tasks to run. Check --sparse-root and its split structure.")
        return

    log("🏁 Start processing with {} workers...".format(args.workers))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (split, drive_dir, cam, input_cam_dir) in enumerate(entries, 1):
            log("  [{:03d}/{:03d}] Queue: {} {} ({})".format(i, len(entries), drive_dir, cam, split))
            futures.append(
                ex.submit(
                    process_cam_dir,
                    args.raw_root, drive_dir, cam, input_cam_dir,
                    args.fill_type, args.extrapolate, args.blur_type, args.show_process,
                    skip_existing=(not args.overwrite),
                    progress_every=args.progress_every
                )
            )

        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
                log("✔️  Task {}/{} finished.".format(i, len(futures)))
            except Exception as e:
                log("⚠️ Task {} failed with exception:\n   {}".format(i, e))

    log("🎉 All done! Dens PNGs are written into <raw_root>/<date>/<drive>/<cam>/dens/.")

if __name__ == "__main__":
    main()
