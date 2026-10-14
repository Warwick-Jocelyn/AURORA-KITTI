import os
import re
import glob
import shutil


def parse_from_val_dens_fname(fname: str):
    """
    支持类似：
    2011_09_26_drive_0002_sync_velodyne_raw_0000000006_image_02.png
    也尽量兼容中间字段变化，只要末尾有 frame(10位) + image_0X
    """
    stem = os.path.splitext(os.path.basename(fname))[0]

    # Regex: date_drive_XXXX_sync ... frame(10d) _image_0X
    m = re.search(
        r"(?P<date>\d{4}_\d{2}_\d{2})_drive_(?P<drive>\d{4})_sync.*?(?P<frame>\d{10})_image_(?P<cam>\d{2})$",
        stem
    )
    if not m:
        raise ValueError(f"Cannot parse val dens filename: {fname}")

    return m.group("date"), m.group("drive"), m.group("frame"), m.group("cam")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def copy_or_link(src: str, dst: str, mode: str = "copy", overwrite: bool = False):
    """
    mode: "copy" or "symlink"
    """
    if (not overwrite) and os.path.exists(dst):
        return

    if overwrite and os.path.exists(dst):
        os.remove(dst)

    if mode == "symlink":
        os.symlink(src, dst)
    else:
        # copy2 keeps metadata
        shutil.copy2(src, dst)


def main():
    data_root = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti"
    gt_root   = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC"

    val_root = os.path.join(data_root, "val_selection_cropped")
    val_dens_dir = os.path.join(val_root, "dens")

    # 输出目录（你说的 dens_clean/image_clean/disp_clean）
    out_dens_clean = os.path.join(val_root, "dens_clean")
    out_disp_clean = os.path.join(val_root, "disp_clean")
    out_img_clean  = os.path.join(val_root, "image_clean")

    ensure_dir(out_dens_clean)
    ensure_dir(out_disp_clean)
    ensure_dir(out_img_clean)

    # 选择 copy 或 symlink（软链接更省空间）
    save_mode = "copy"       # 改成 "symlink" 也可以
    overwrite = False        # True 会覆盖已存在文件

    dens_files = sorted(glob.glob(os.path.join(val_dens_dir, "*.png")))
    print("val dens files:", len(dens_files))
    if not dens_files:
        print("[ERR] No dens png found in:", val_dens_dir)
        return

    n_ok = 0
    missing = []

    for dens_path in dens_files:
        dens_fname = os.path.basename(dens_path)

        try:
            date, drive, frame, cam = parse_from_val_dens_fname(dens_fname)
        except Exception as e:
            missing.append((dens_fname, f"parse_failed: {e}"))
            continue

        # gt_root/raw mapping
        cam_root = os.path.join(
            gt_root, "raw",
            date, f"{date}_drive_{drive}_sync",
            f"image_{cam}"
        )

        clean_dens_src = os.path.join(cam_root, "dens", f"{frame}.png")
        clean_disp_src = os.path.join(cam_root, "disp", f"{frame}.png")
        clean_rgb_src  = os.path.join(cam_root, "data", f"{frame}.png")

        # 目标文件名：为了和你 val 的命名对齐
        # - dens_clean 用原 dens_fname（velodyne_raw_...）
        # - disp_clean / image_clean 用 disp_fname（把 velodyne_raw_ -> image_）
        disp_fname = dens_fname.replace("velodyne_raw_", "image_")

        clean_dens_dst = os.path.join(out_dens_clean, dens_fname)
        clean_disp_dst = os.path.join(out_disp_clean, disp_fname)
        clean_rgb_dst  = os.path.join(out_img_clean,  disp_fname)

        ok = True
        for p in [clean_dens_src, clean_disp_src, clean_rgb_src]:
            if not os.path.exists(p):
                ok = False
                missing.append((dens_fname, f"missing_src: {p}"))
                break
        if not ok:
            continue

        copy_or_link(clean_dens_src, clean_dens_dst, mode=save_mode, overwrite=overwrite)
        copy_or_link(clean_disp_src, clean_disp_dst, mode=save_mode, overwrite=overwrite)
        copy_or_link(clean_rgb_src,  clean_rgb_dst,  mode=save_mode, overwrite=overwrite)

        n_ok += 1

        if n_ok <= 3:
            print("\n[Example OK]")
            print(" val dens:", dens_path)
            print(" clean dens src -> dst:", clean_dens_src, "->", clean_dens_dst)
            print(" clean disp src -> dst:", clean_disp_src, "->", clean_disp_dst)
            print(" clean rgb  src -> dst:", clean_rgb_src,  "->", clean_rgb_dst)

    print("\n=== SUMMARY ===")
    print("found & saved:", n_ok, "/", len(dens_files))
    print("missing:", len(missing))
    if missing[:10]:
        print("\nFirst 10 missing:")
        for item in missing[:10]:
            print(" -", item[0], "|", item[1])


if __name__ == "__main__":
    main()
