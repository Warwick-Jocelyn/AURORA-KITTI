import os
import re
import glob
import shutil
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


# =============== 尺寸配置 ===============
RESIZE_W, RESIZE_H = 1242, 375   # (width, height)
CROP_W, CROP_H     = 1216, 352   # (width, height)
# ======================================


def parse_from_val_dens_fname(fname: str):
    """
    支持类似：
    2011_09_26_drive_0002_sync_velodyne_raw_0000000006_image_02.png
    Regex: date_drive_XXXX_sync ... frame(10d) _image_0X
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(
        r"(?P<date>\d{4}_\d{2}_\d{2})_drive_(?P<drive>\d{4})_sync.*?(?P<frame>\d{10})_image_(?P<cam>\d{2})$",
        stem
    )
    if not m:
        raise ValueError(f"Cannot parse val dens filename: {fname}")
    return m.group("date"), m.group("drive"), m.group("frame"), m.group("cam")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str = "copy", overwrite: bool = False):
    """
    mode: "copy" or "symlink"
    """
    if (not overwrite) and dst.exists():
        return

    if overwrite and dst.exists():
        dst.unlink()

    ensure_dir(dst.parent)

    if mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def compute_crop_box(w, h, target_w, target_h):
    """
    Center-Bottom crop:
    - 高度：保留底部，裁掉上方 => top = h - target_h
    - 宽度：居中
    """
    if w < target_w or h < target_h:
        return None

    top = h - target_h
    bottom = h

    to_crop_w = w - target_w
    left = to_crop_w // 2
    right_extra = to_crop_w - left
    right = w - right_extra

    # clamp
    left = max(0, min(left, w - target_w))
    right = max(left + target_w, min(w, right))
    top = max(0, min(top, h - target_h))
    bottom = max(top + target_h, min(h, bottom))

    return left, top, right, bottom


def resize_image_cv2(img, dst_w, dst_h, interpolation):
    # cv2.resize size=(width,height)
    return cv2.resize(img, (dst_w, dst_h), interpolation=interpolation)


def crop_image_np(img, crop_w, crop_h):
    """
    img: numpy array (H,W) or (H,W,C)
    """
    h, w = img.shape[:2]
    box = compute_crop_box(w, h, crop_w, crop_h)
    if box is None:
        return None
    left, top, right, bottom = box
    return img[top:bottom, left:right]


def read_dens_16u(path: Path):
    """
    读取 16-bit 单通道 dens
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read dens: {path}")
    if arr.dtype != "uint16":
        raise RuntimeError(f"dens is not uint16: {path}, dtype={arr.dtype}")
    if arr.ndim != 2:
        raise RuntimeError(f"dens is not single-channel: {path}, shape={arr.shape}")
    return arr


def read_disp(path: Path):
    """
    disp: 可能是 8-bit 单通道，也可能别的；按 UNCHANGED 读，保持通道/bit-depth
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read disp: {path}")
    return arr


def read_rgb(path: Path):
    """
    RGB: 用 COLOR 读，得到 8-bit 3通道（BGR存储，但写回png/jpg没问题）
    """
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"Cannot read rgb: {path}")
    return arr


def write_image(path: Path, arr):
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")


def process_one_triplet(
    clean_dens_src: Path,
    clean_disp_src: Path,
    clean_rgb_src: Path,
    dens_fname: str,
    disp_fname: str,
    out_root: Path,
    save_mode: str = "copy",
    overwrite: bool = False,
):
    """
    输出目录结构（都在 out_root 下）：
      dens_clean_original / dens_clean_resized / dens_clean
      disp_clean_original / disp_clean_resized / disp_clean
      image_clean_original / image_clean_resized / image_clean
    """

    # --- Stage 0: copy/link clean originals ---
    dens_ori = out_root / "dens_clean_original" / dens_fname
    disp_ori = out_root / "disp_clean_original" / disp_fname
    rgb_ori  = out_root / "image_clean_original" / disp_fname

    copy_or_link(clean_dens_src, dens_ori, mode=save_mode, overwrite=overwrite)
    copy_or_link(clean_disp_src, disp_ori, mode=save_mode, overwrite=overwrite)
    copy_or_link(clean_rgb_src,  rgb_ori,  mode=save_mode, overwrite=overwrite)

    # --- Stage 1: resize to 1242x375 ---
    dens_res = out_root / "dens_clean_resized" / dens_fname
    disp_res = out_root / "disp_clean_resized" / disp_fname
    rgb_res  = out_root / "image_clean_resized" / disp_fname

    # dens resize (uint16, nearest)
    if overwrite or (not dens_res.exists()):
        d = read_dens_16u(dens_ori)
        h, w = d.shape[:2]
        if (w, h) == (RESIZE_W, RESIZE_H):
            shutil.copy2(str(dens_ori), str(dens_res))
        else:
            d2 = resize_image_cv2(d, RESIZE_W, RESIZE_H, interpolation=cv2.INTER_NEAREST)
            write_image(dens_res, d2)

    # disp resize (area)
    if overwrite or (not disp_res.exists()):
        dp = read_disp(disp_ori)
        h, w = dp.shape[:2]
        if (w, h) == (RESIZE_W, RESIZE_H):
            shutil.copy2(str(disp_ori), str(disp_res))
        else:
            dp2 = resize_image_cv2(dp, RESIZE_W, RESIZE_H, interpolation=cv2.INTER_AREA)
            write_image(disp_res, dp2)

    # rgb resize (area)
    if overwrite or (not rgb_res.exists()):
        im = read_rgb(rgb_ori)
        h, w = im.shape[:2]
        if (w, h) == (RESIZE_W, RESIZE_H):
            shutil.copy2(str(rgb_ori), str(rgb_res))
        else:
            im2 = resize_image_cv2(im, RESIZE_W, RESIZE_H, interpolation=cv2.INTER_AREA)
            write_image(rgb_res, im2)

    # --- Stage 2: crop to 1216x352 (cut top + center width, keep bottom) ---
    dens_out = out_root / "dens_clean" / dens_fname
    disp_out = out_root / "disp_clean" / disp_fname
    rgb_out  = out_root / "image_clean" / disp_fname

    # dens crop
    if overwrite or (not dens_out.exists()):
        d = read_dens_16u(dens_res)
        cropped = crop_image_np(d, CROP_W, CROP_H)
        if cropped is None:
            raise RuntimeError(f"dens too small to crop: {dens_res} size={d.shape[::-1]}")
        write_image(dens_out, cropped)

    # disp crop
    if overwrite or (not disp_out.exists()):
        dp = read_disp(disp_res)
        cropped = crop_image_np(dp, CROP_W, CROP_H)
        if cropped is None:
            raise RuntimeError(f"disp too small to crop: {disp_res} size={dp.shape[:2]}")
        write_image(disp_out, cropped)

    # rgb crop
    if overwrite or (not rgb_out.exists()):
        im = read_rgb(rgb_res)
        cropped = crop_image_np(im, CROP_W, CROP_H)
        if cropped is None:
            raise RuntimeError(f"rgb too small to crop: {rgb_res} size={im.shape[:2]}")
        write_image(rgb_out, cropped)


def main():
    parser = argparse.ArgumentParser("Copy clean dens/disp/image -> resize(1242x375) -> crop(1216x352)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="例如: /.../datas/kitti")
    parser.add_argument("--gt_root", type=str, required=True,
                        help="例如: /.../datas/kitti_depth_DC")
    parser.add_argument("--val_name", type=str, default="val_selection_cropped",
                        help="val folder name, 默认 val_selection_cropped")
    parser.add_argument("--save_mode", choices=["copy", "symlink"], default="copy",
                        help="原始 clean 文件保存方式：copy 或 symlink")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    gt_root   = Path(args.gt_root)

    val_root = data_root / args.val_name
    val_dens_dir = val_root / "dens"

    if not val_dens_dir.exists():
        raise FileNotFoundError(f"Not found: {val_dens_dir}")

    # 创建输出目录（全部在 val_root 下）
    for d in [
        "dens_clean_original", "disp_clean_original", "image_clean_original",
        "dens_clean_resized",  "disp_clean_resized",  "image_clean_resized",
        "dens_clean",          "disp_clean",          "image_clean",
    ]:
        ensure_dir(val_root / d)

    dens_files = sorted(glob.glob(str(val_dens_dir / "*.png")))
    print("val dens files:", len(dens_files))
    if not dens_files:
        print("[ERR] No dens png found in:", val_dens_dir)
        return

    n_ok = 0
    missing = []
    failed = []

    for dens_path in tqdm(dens_files, desc="All triplets"):
        dens_path = Path(dens_path)
        dens_fname = dens_path.name

        try:
            date, drive, frame, cam = parse_from_val_dens_fname(dens_fname)
        except Exception as e:
            missing.append((dens_fname, f"parse_failed: {e}"))
            continue

        cam_root = gt_root / "raw" / date / f"{date}_drive_{drive}_sync" / f"image_{cam}"

        clean_dens_src = cam_root / "dens" / f"{frame}.png"
        clean_disp_src = cam_root / "disp" / f"{frame}.png"
        clean_rgb_src  = cam_root / "data" / f"{frame}.png"

        disp_fname = dens_fname.replace("velodyne_raw_", "image_")

        # 检查源是否存在
        ok = True
        for p in [clean_dens_src, clean_disp_src, clean_rgb_src]:
            if not p.exists():
                ok = False
                missing.append((dens_fname, f"missing_src: {p}"))
                break
        if not ok:
            continue

        try:
            process_one_triplet(
                clean_dens_src=clean_dens_src,
                clean_disp_src=clean_disp_src,
                clean_rgb_src=clean_rgb_src,
                dens_fname=dens_fname,
                disp_fname=disp_fname,
                out_root=val_root,
                save_mode=args.save_mode,
                overwrite=args.overwrite,
            )
            n_ok += 1
        except Exception as e:
            failed.append((dens_fname, str(e)))

    print("\n=== SUMMARY ===")
    print("OK:", n_ok, "/", len(dens_files))
    print("missing:", len(missing))
    print("failed:", len(failed))

    if missing[:10]:
        print("\nFirst 10 missing:")
        for a, b in missing[:10]:
            print(" -", a, "|", b)

    if failed[:10]:
        print("\nFirst 10 failed:")
        for a, b in failed[:10]:
            print(" -", a, "|", b)

    print("\nDone.")
    print("Outputs in:", val_root)


if __name__ == "__main__":
    main()


# python make_val_clean_resize_crop.py \
#   --data_root /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti \
#   --gt_root   /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC \
#   --val_name  val_selection_cropped \
#   --save_mode copy
# --overwrite