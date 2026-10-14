#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
from tqdm import tqdm

# ===== 配置区 =====
SOURCES = {
    "16bit_proj_depth_Clean": Path("/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset/16bit_proj_depth_Clean"),
    "16bit_proj_depth_fog":   Path("/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset/16bit_proj_depth_fog"),
    "16bit_proj_depth_Rain":  Path("/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset/16bit_proj_depth_Rain"),
    "16bit_proj_depth_Snow":  Path("/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset/16bit_proj_depth_Snow"),
}

# 目标根目录：扁平化后的存放位置
DEST_ROOT = Path("/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset/16bit_proj_depth")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".exr"}
SPLITS = ["train", "val", "test"]

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def list_image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if is_image_file(p)]

def main():
    all_files = []
    for split in SPLITS:
        for src_root in SOURCES.values():
            split_dir = src_root / split
            if split_dir.exists():
                files = list_image_files(split_dir)
                dest_dir = DEST_ROOT / split
                all_files.extend((f, dest_dir) for f in files)

    if not all_files:
        print("未找到需要复制的图像文件，请检查路径和扩展名。")
        return

    # 创建目标目录结构
    for split in SPLITS:
        (DEST_ROOT / split).mkdir(parents=True, exist_ok=True)

    # 复制
    with tqdm(total=len(all_files), desc="Copying images", unit="file") as pbar:
        for src_file, dest_dir in all_files:
            target_path = dest_dir / src_file.name

            # 如果有同名文件冲突，加上序号避免覆盖
            if target_path.exists():
                stem, suffix = src_file.stem, src_file.suffix
                i = 1
                while True:
                    new_name = f"{stem}_{i}{suffix}"
                    new_target = dest_dir / new_name
                    if not new_target.exists():
                        target_path = new_target
                        break
                    i += 1

            shutil.copy2(src_file, target_path)
            pbar.update(1)

    print(f"完成！所有文件已复制到 {DEST_ROOT}")

if __name__ == "__main__":
    main()
