#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path


def collect_stems(dir_path: Path, suffixes: set[str], recursive: bool) -> set[str]:
    """收集目录中指定后缀文件的 stem 集合。"""
    if recursive:
        files = (p for p in dir_path.rglob("*") if p.is_file())
    else:
        files = (p for p in dir_path.glob("*") if p.is_file())

    stems = set()
    for p in files:
        if p.suffix.lower() in suffixes:
            stems.add(p.stem)
    return stems


def iter_files(dir_path: Path, suffixes: set[str], recursive: bool):
    """遍历目录中指定后缀的文件。"""
    if recursive:
        it = (p for p in dir_path.rglob("*") if p.is_file())
    else:
        it = (p for p in dir_path.glob("*") if p.is_file())

    for p in it:
        if p.suffix.lower() in suffixes:
            yield p


def main():
    ap = argparse.ArgumentParser(
        description=(
            "根据 velodyne_raw 里存在的 png 文件名，过滤 velodyne_raw_npy_in_meter 里的 npy。\n"
            "规则：若 npy 的 stem 在 velodyne_raw 中找不到同名 png，则删除/移动该 npy。"
        )
    )
    ap.add_argument(
        "--velodyne_png_dir",
        type=str,
        required=True,
        help="velodyne_raw 目录（包含 .png 的那个目录）",
    )
    ap.add_argument(
        "--npy_dir",
        type=str,
        required=True,
        help="velodyne_raw_npy_in_meter 目录（包含 .npy 的那个目录）",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录（如果你的数据在子目录里就打开）",
    )

    # 安全选项
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要删除/移动的文件，不实际操作",
    )
    ap.add_argument(
        "--move_to",
        type=str,
        default="",
        help="不直接删除，而是把要删的 npy 移动到该目录（建议先用这个）",
    )

    # 如果你的 velodyne_raw 不是 png，也可以扩展
    ap.add_argument(
        "--png_suffixes",
        type=str,
        default=".png",
        help="作为白名单的后缀（逗号分隔），默认仅 .png，比如：.png,.jpg",
    )

    args = ap.parse_args()

    velodyne_png_dir = Path(args.velodyne_png_dir)
    npy_dir = Path(args.npy_dir)
    recursive = args.recursive
    dry_run = args.dry_run
    move_to = Path(args.move_to) if args.move_to else None

    if not velodyne_png_dir.exists():
        raise FileNotFoundError(f"velodyne_png_dir 不存在: {velodyne_png_dir}")
    if not npy_dir.exists():
        raise FileNotFoundError(f"npy_dir 不存在: {npy_dir}")

    png_suffixes = {s.strip().lower() for s in args.png_suffixes.split(",") if s.strip()}
    if not png_suffixes:
        raise ValueError("png_suffixes 解析为空，请检查 --png_suffixes 参数")

    # 1) 收集 velodyne_raw 里存在的 png stems（白名单）
    whitelist_stems = collect_stems(velodyne_png_dir, png_suffixes, recursive)
    if len(whitelist_stems) == 0:
        raise RuntimeError(f"在 {velodyne_png_dir} 没找到后缀为 {png_suffixes} 的文件")

    print(f"[INFO] 白名单 stems 数量（来自 velodyne_raw）: {len(whitelist_stems)}")

    # 2) 遍历 npy_dir 下所有 npy，若 stem 不在白名单，则删除/移动
    npy_suffixes = {".npy"}
    total_npy = 0
    to_remove = []

    for npy_path in iter_files(npy_dir, npy_suffixes, recursive):
        total_npy += 1
        if npy_path.stem not in whitelist_stems:
            to_remove.append(npy_path)

    print(f"[INFO] 扫描到 npy 总数: {total_npy}")
    print(f"[INFO] 需要删除/移动的 npy 数量（在 velodyne_raw 找不到同名 png）: {len(to_remove)}")

    # 打印前若干个样例
    preview_n = 30
    if to_remove:
        print(f"\n[PREVIEW] 前 {min(preview_n, len(to_remove))} 个将被处理的文件：")
        for p in to_remove[:preview_n]:
            print("  ", p)

    if dry_run:
        print("\n[DRY RUN] 已开启 dry_run，不会进行删除/移动。")
        return

    # 3) 执行：移动或删除
    if move_to:
        move_to.mkdir(parents=True, exist_ok=True)
        moved = 0
        for src in to_remove:
            dst = move_to / src.name  # 平铺移动（文件名通常唯一）
            # 若冲突，追加一个后缀避免覆盖
            if dst.exists():
                i = 1
                while True:
                    alt = move_to / f"{src.stem}__dup{i}{src.suffix}"
                    if not alt.exists():
                        dst = alt
                        break
                    i += 1
            shutil.move(str(src), str(dst))
            moved += 1
        print(f"\n[DONE] 已移动 {moved} 个 npy 到: {move_to}")
    else:
        deleted = 0
        for p in to_remove:
            p.unlink()
            deleted += 1
        print(f"\n[DONE] 已删除 {deleted} 个 npy（直接 unlink）")

    print("[DONE] 完成。")


if __name__ == "__main__":
    main()

# python filter_velodyne_npy_by_png.py \
#   --velodyne_png_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small/velodyne_raw \
#   --npy_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small/velodyne_raw_npy_in_meter \
#   --move_to /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST_small/velodyne_raw_npy_in_meter__removed
