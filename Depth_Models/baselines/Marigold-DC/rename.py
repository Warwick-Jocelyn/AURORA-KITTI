import os
from pathlib import Path

# 你要替换的字符串
OLD = "_sync_groundtruth_depth_"
NEW = "_sync_velodyne_raw_"

# 常见图像后缀（可按需增删）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def rename_images(folder: str, recursive: bool = False, dry_run: bool = False) -> None:
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise ValueError(f"不是有效文件夹: {folder}")

    pattern = "**/*" if recursive else "*"

    changed = 0
    skipped = 0
    conflicts = 0

    for p in base.glob(pattern):
        if not p.is_file():
            continue

        # 仅处理图像文件
        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        name = p.name
        if OLD not in name:
            skipped += 1
            continue

        new_name = name.replace(OLD, NEW)
        new_path = p.with_name(new_name)

        # 重名冲突处理：目标已存在则跳过
        if new_path.exists():
            conflicts += 1
            print(f"[冲突跳过] {p} -> {new_path} (目标已存在)")
            continue

        print(f"[重命名] {p.name} -> {new_name}")
        if not dry_run:
            p.rename(new_path)
        changed += 1

    print("\n=== 结果 ===")
    print(f"已重命名: {changed}")
    print(f"无匹配跳过: {skipped}")
    print(f"重名冲突跳过: {conflicts}")

if __name__ == "__main__":
    # 修改成你的文件夹路径
    folder_path = r"/path/to/aurora-root/Code/Marigold-DC/Marigold_DC_result_16bit_Aurora_test"

    # recursive=True 会递归子文件夹；dry_run=True 只预览不真正改名
    rename_images(folder_path, recursive=False, dry_run=False)
