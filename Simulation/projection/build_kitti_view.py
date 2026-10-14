# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# from pathlib import Path
# import re
# import shutil

# # ------------- 配置 -------------
# # 源数据（你的扁平化数据根）
# SRC_ROOT = Path("/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/Split_Flattened_Dataset")

# # 目标视图（生成 KITTI-like 的根，不会复制大文件，只建软链）
# DST_ROOT = Path("/path/to/aurora-root/Dataset/AuroraKITTI")

# # 支持的 split
# SPLITS = ["train", "val", "test"]  # 按需调整

# # 是否用软链接（True）或复制（False）
# USE_SYMLINK = True

# # ------------- 命名解析 -------------
# # RGB 文件名示例：
# # 2011_09_26_drive_0002_sync_image_02_0000000006.png
# PAT_RGB = re.compile(
#     r'(?P<seq>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<cam>\d{2})_(?P<frame>\d{10})\.png$'
# )

# # 稀疏深度文件名示例：
# # 2011_09_26_drive_0002_sync_proj_depth_02_0000000006.png
# PAT_SPARSE = re.compile(
#     r'(?P<seq>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_proj_depth_(?P<cam>\d{2})_(?P<frame>\d{10})\.png$'
# )

# # GT 文件名示例：
# # 2011_09_26_drive_0002_sync_depthGT_02_0000000006.png
# PAT_GT = re.compile(
#     r'(?P<seq>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_depthGT_(?P<cam>\d{2})_(?P<frame>\d{10})\.png$'
# )

# def safe_link_or_copy(src: Path, dst: Path):
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists() or dst.is_symlink():
#         dst.unlink()
#     if USE_SYMLINK:
#         # 在 Windows 可能需要管理员权限；失败就退化为复制
#         try:
#             os.symlink(src.resolve(), dst)
#         except Exception as e:
#             print(f"[warn] symlink failed ({e}), fallback to copy: {src} -> {dst}")
#             shutil.copy2(src, dst)
#     else:
#         shutil.copy2(src, dst)

# def index_dir(d: Path, pattern: re.Pattern):
#     """返回 { (seq, cam_str, frame): Path }"""
#     out = {}
#     for p in d.glob("*.png"):
#         m = pattern.search(p.name)
#         if not m:
#             continue
#         seq = m.group("seq")
#         cam = m.group("cam")  # '02' / '03'
#         frame = m.group("frame")
#         key = (seq, cam, frame)
#         out[key] = p
#     return out

# def main():
#     # 源三个模态的父目录
#     rgb_root = SRC_ROOT / "frame_image"
#     sparse_root = SRC_ROOT / "16bit_proj_depth"
#     gt_root = SRC_ROOT / "depth_GT"

#     total_links = 0
#     missing_report = []

#     for split in SPLITS:
#         rgb_dir = rgb_root / split
#         sparse_dir = sparse_root / split
#         gt_dir = gt_root / split

#         # 建立索引
#         rgb_map = index_dir(rgb_dir, PAT_RGB) if rgb_dir.exists() else {}
#         sparse_map = index_dir(sparse_dir, PAT_SPARSE) if sparse_dir.exists() else {}
#         gt_map = index_dir(gt_dir, PAT_GT) if gt_dir.exists() else {}

#         # 用联合键遍历（谁有就转谁；RGB 必须尽量覆盖 d / gt）
#         all_keys = set(rgb_map.keys()) | set(sparse_map.keys()) | set(gt_map.keys())

#         for (seq, cam, frame) in sorted(all_keys):
#             cam_dir = f"image_{cam}"  # 'image_02' / 'image_03'

#             # ---------- RGB ----------
#             src = rgb_map.get((seq, cam, frame))
#             if src is not None:
#                 dst = DST_ROOT / "data_rgb" / split / seq / cam_dir / "data" / f"{frame}.png"
#                 safe_link_or_copy(src, dst)
#                 total_links += 1
#             else:
#                 # 如果缺 RGB，但存在 d 或 gt，会导致 dataloader 里 get_rgb_paths 找不到
#                 if ((seq, cam, frame) in sparse_map) or ((seq, cam, frame) in gt_map):
#                     missing_report.append(("RGB", split, seq, cam, frame))

#             # ---------- Sparse depth ----------
#             src = sparse_map.get((seq, cam, frame))
#             if src is not None:
#                 dst = DST_ROOT / "data_depth_velodyne" / split / seq / "proj_depth" / "velodyne_raw" / cam_dir / f"{frame}.png"
#                 safe_link_or_copy(src, dst)
#                 total_links += 1

#             # ---------- GT ----------
#             src = gt_map.get((seq, cam, frame))
#             if src is not None:
#                 dst = DST_ROOT / "data_depth_annotated" / split / seq / "proj_depth" / "groundtruth" / cam_dir / f"{frame}.png"
#                 safe_link_or_copy(src, dst)
#                 total_links += 1

#     print(f"[done] created {total_links} links/files under: {DST_ROOT}")
#     if missing_report:
#         print("[warn] samples missing RGB but having d/gt:")
#         for rec in missing_report[:50]:
#             print("  missing", rec)
#         if len(missing_report) > 50:
#             print(f"  ... and {len(missing_report)-50} more")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import re
import shutil

# ------------- 配置 -------------
SRC_ROOT = Path("/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/Split_Flattened_Dataset")
DST_ROOT = Path("/path/to/aurora-root/Dataset/AuroraKITTI")

SPLITS = ["train", "val", "test"]
USE_SYMLINK = True

# ------------- 命名解析 -------------
# intensity 文件名示例：
# 2011_09_26_drive_0002_sync_proj_intensity_02_0000000006.png
PAT_INTENSITY = re.compile(
    r'(?P<seq>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_proj_intenisty_(?P<cam>\d{2})_(?P<frame>\d{10})\.png$'
)

# height 文件名示例：
# 2011_09_26_drive_0002_sync_proj_height_02_0000000006.png
PAT_HEIGHT = re.compile(
    r'(?P<seq>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_proj_height_(?P<cam>\d{2})_(?P<frame>\d{10})\.png$'
)

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if USE_SYMLINK:
        try:
            os.symlink(src.resolve(), dst)
        except Exception as e:
            print(f"[warn] symlink failed ({e}), fallback to copy: {src} -> {dst}")
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def index_dir(d: Path, pattern: re.Pattern):
    """返回 { (seq, cam, frame): Path }"""
    out = {}
    for p in d.glob("*.png"):
        m = pattern.search(p.name)
        if not m:
            continue
        seq = m.group("seq")
        cam = m.group("cam")
        frame = m.group("frame")
        key = (seq, cam, frame)
        out[key] = p
    return out

def main():
    intensity_root = SRC_ROOT / "intenisty"
    height_root = SRC_ROOT / "height"

    total_links = 0

    for split in SPLITS:
        inten_dir = intensity_root / split
        height_dir = height_root / split

        intensity_map = index_dir(inten_dir, PAT_INTENSITY) if inten_dir.exists() else {}
        height_map = index_dir(height_dir, PAT_HEIGHT) if height_dir.exists() else {}

        all_keys = set(intensity_map.keys()) | set(height_map.keys())

        for (seq, cam, frame) in sorted(all_keys):
            cam_dir = f"image_{cam}"

            # ---------- Intensity ----------
            src = intensity_map.get((seq, cam, frame))
            if src is not None:
                dst = DST_ROOT / "data_intensity_velodyne" / split / seq / cam_dir / "data" / f"{frame}.png"
                safe_link_or_copy(src, dst)
                total_links += 1

            # ---------- Height ----------
            src = height_map.get((seq, cam, frame))
            if src is not None:
                dst = DST_ROOT / "data_height_velodyne" / split / seq / cam_dir / "data" / f"{frame}.png"
                safe_link_or_copy(src, dst)
                total_links += 1

    print(f"[done] created {total_links} links/files under: {DST_ROOT}")

if __name__ == "__main__":
    main()
