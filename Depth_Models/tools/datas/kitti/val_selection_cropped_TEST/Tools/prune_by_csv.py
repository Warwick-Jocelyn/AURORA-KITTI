# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# import pandas as pd


# def parse_csv_name(name: str):
#     """
#     CSV 的 name 示例：
#       2011_09_26_drive_0011_sync_image_02_0000000005

#     返回：
#       (sequence_id, frame_str_10, cam_str)  e.g.
#       ("2011_09_26_drive_0011_sync", "0000000005", "image_02")
#     """
#     parts = name.split("_")
#     if len(parts) < 4:
#         raise ValueError(f"CSV name 格式异常: {name}")

#     frame = parts[-1]
#     cam = "_".join(parts[-3:-1])      # image_02
#     seq = "_".join(parts[:-3])        # 2011_09_26_drive_0011_sync
#     frame = frame.zfill(10)
#     return seq, frame, cam


# def build_expected_filenames_from_csv(df: pd.DataFrame):
#     """
#     根据 CSV 的 sequence_id + name，生成每个文件夹中应该存在的文件名集合。
#     """
#     expected = {
#         "dens": set(),
#         "disp": set(),
#         "groundtruth_depth": set(),
#         "velodyne_raw": set(),
#         "intrinsics": set(),
#     }

#     for _, row in df.iterrows():
#         seq = str(row["sequence_id"])
#         name = str(row["name"])
#         seq2, frame, cam = parse_csv_name(name)

#         # 如果 CSV 里 sequence_id 和 name 解析出来的 seq 不一致，仍以 CSV 的 sequence_id 为准，但给个提示
#         if seq2 != seq:
#             # 不中断，只用 CSV sequence_id
#             seq2 = seq

#         # 这些命名规则按你举例来的
#         expected["dens"].add(f"{seq2}_velodyne_raw_{frame}_{cam}.png")
#         expected["velodyne_raw"].add(f"{seq2}_velodyne_raw_{frame}_{cam}.png")

#         expected["disp"].add(f"{seq2}_image_{frame}_{cam}.png")
#         expected["groundtruth_depth"].add(f"{seq2}_groundtruth_depth_{frame}_{cam}.png")
#         expected["intrinsics"].add(f"{seq2}_image_{frame}_{cam}.txt")

#     return expected


# def prune_folder(folder: Path, keep_names: set, dry_run: bool, exts: set):
#     """
#     删除 folder 下不在 keep_names 里的文件（仅针对给定后缀 exts）。
#     """
#     folder.mkdir(parents=True, exist_ok=True)
#     files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

#     to_delete = [p for p in files if p.name not in keep_names]
#     keep_exist = [p for p in files if p.name in keep_names]

#     # 也统计一下“CSV 要求保留但文件缺失”的情况
#     existing_names = {p.name for p in files}
#     missing = sorted(list(keep_names - existing_names))

#     print(f"\n=== Folder: {folder} ===")
#     print(f"Found files: {len(files)}")
#     print(f"Will keep : {len(keep_exist)}")
#     print(f"Will delete: {len(to_delete)}")
#     print(f"Missing (expected by CSV but not found on disk): {len(missing)}")

#     if missing[:10]:
#         print("  Missing examples:")
#         for x in missing[:10]:
#             print("   -", x)

#     if to_delete[:10]:
#         print("  Delete examples:")
#         for p in to_delete[:10]:
#             print("   -", p.name)

#     if not dry_run:
#         for p in to_delete:
#             try:
#                 p.unlink()
#             except Exception as e:
#                 print(f"[WARN] 删除失败: {p}  -> {e}")

#     return len(files), len(keep_exist), len(to_delete), len(missing)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv_small", type=str, required=True, help="更新后的 CSV（例如 test_small.csv）")
#     ap.add_argument("--base_dir", type=str, required=True, help="val_selection_cropped_TEST_small 的路径")
#     ap.add_argument("--apply", action="store_true", help="真正执行删除（默认 dry-run 只打印）")
#     args = ap.parse_args()

#     csv_path = Path(args.csv_small)
#     base_dir = Path(args.base_dir)

#     if not csv_path.exists():
#         raise FileNotFoundError(f"CSV 不存在: {csv_path}")
#     if not base_dir.exists():
#         raise FileNotFoundError(f"base_dir 不存在: {base_dir}")

#     df = pd.read_csv(csv_path)
#     for col in ["sequence_id", "name"]:
#         if col not in df.columns:
#             raise ValueError(f"CSV 缺少必要列: {col}")

#     print(f"CSV rows: {len(df)}")
#     expected = build_expected_filenames_from_csv(df)

#     dry_run = (not args.apply)
#     if dry_run:
#         print("\n[DRY-RUN] 仅显示将删除的文件，不会真的删除。要执行删除请加 --apply")
#     else:
#         print("\n[APPLY] 将真正删除不需要的文件！")

#     # 每个文件夹对应的后缀
#     folder_specs = {
#         "dens": {".png"},
#         "disp": {".png"},
#         "groundtruth_depth": {".png"},
#         "velodyne_raw": {".png"},
#         "intrinsics": {".txt"},
#     }

#     total_found = total_keep = total_del = total_missing = 0
#     for sub, exts in folder_specs.items():
#         folder = base_dir / sub
#         found, kept, deleted, missing = prune_folder(
#             folder=folder,
#             keep_names=expected[sub],
#             dry_run=dry_run,
#             exts=exts,
#         )
#         total_found += found
#         total_keep += kept
#         total_del += deleted
#         total_missing += missing

#     print("\n========== SUMMARY ==========")
#     print(f"Total found   : {total_found}")
#     print(f"Total keep    : {total_keep}")
#     print(f"Total delete  : {total_del}")
#     print(f"Total missing : {total_missing}")
#     print("Done.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import shutil
import os
from pathlib import Path
import pandas as pd


def parse_image_stem_to_seq_and_name(stem: str):
    """
    支持两种常见格式：
    1) {sequence_id}_image_{frame}_image_02  -> name={sequence_id}_image_02_{frame}
    2) {sequence_id}_image_02_{frame}        -> name={sequence_id}_image_02_{frame}
    frame 最终 zfill(10)
    """
    m = re.match(r"(.+)_image_(\d+)_((?:image)_\d{2})$", stem)
    if m:
        seq, frame, cam = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    m = re.match(r"(.+)_((?:image)_\d{2})_(\d+)$", stem)
    if m:
        seq, cam, frame = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    raise ValueError(f"无法解析图像名: {stem}")


def parse_velodyne_stem_to_seq_and_name(stem: str):
    """
    支持 velodyne 常见格式（按你的例子）：
    1) {sequence_id}_velodyne_raw_{frame}_image_02
       -> seq={sequence_id}, name={sequence_id}_image_02_{frame}
    2) {sequence_id}_velodyne_raw_{frame}_{image_02}  (与1等价，只是写法更宽松)

    如果你后面还有别的 velodyne 命名（比如 velodyne_points / velodyne_raw_npy 等），
    你可以在这里继续加 regex。
    """
    m = re.match(r"(.+)_velodyne_raw_(\d+)_((?:image)_\d{2})$", stem)
    if m:
        seq, frame, cam = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    # 兜底：允许末尾 camera 写成 image_02，但中间多一个下划线等
    m = re.match(r"(.+)_velodyne_raw_(\d+)_((?:image)_\d{2})$", stem)
    if m:
        seq, frame, cam = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    raise ValueError(f"无法解析 velodyne 名: {stem}")


def stat_counts(series: pd.Series, categories):
    total = int(series.shape[0])
    vc = series.value_counts(dropna=False)
    rows = []
    for cat in categories:
        c = int(vc.get(cat, 0))
        ratio = (c / total) if total > 0 else 0.0
        rows.append((cat, c, ratio))
    return total, rows


def print_stats(title: str, df: pd.DataFrame):
    print(f"\n========== {title} ==========")

    total_w, rows_w = stat_counts(df["weather"], ["Clear", "Fog", "Rain", "Snow"])
    print(f"[weather] total={total_w}")
    for cat, c, r in rows_w:
        print(f"  - {cat:5s}: {c:4d}  ({r*100:6.2f}%)")

    total_t, rows_t = stat_counts(df["Time_of_Day"], ["Day", "Night"])
    print(f"[Time_of_Day] total={total_t}")
    for cat, c, r in rows_t:
        print(f"  - {cat:5s}: {c:4d}  ({r*100:6.2f}%)")


def scan_files(root: Path, exts: set[str], recursive: bool):
    if recursive:
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name)
    return sorted([p for p in root.glob("*") if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name)


def build_index_df(files, parser_fn, path_col: str):
    rows = []
    bad = []
    for p in files:
        try:
            seq, name = parser_fn(p.stem)
            rows.append({"sequence_id": seq, "name": name, path_col: str(p)})
        except Exception:
            bad.append(str(p))
    df = pd.DataFrame(rows).drop_duplicates(subset=["sequence_id", "name"])
    return df, bad


def materialize_filtered_files(matched_df: pd.DataFrame, src_col: str, out_dir: Path, mode: str):
    """
    mode: 'copy' | 'symlink'
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for _, r in matched_df.iterrows():
        src = Path(r[src_col])
        dst = out_dir / src.name  # 直接放平；如需保留子目录结构可再扩展
        if dst.exists():
            # 避免重复冲突（一般不会发生，因为 name 唯一，文件名也通常唯一）
            continue

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            # 软链接：dst 指向 src
            os.symlink(src, dst)
        else:
            raise ValueError(f"未知 mode: {mode}")
        n += 1

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", type=str, required=True, help="原始 test.csv 路径")
    ap.add_argument("--image_dir", type=str, required=True, help="当前 image 文件夹路径")
    ap.add_argument("--csv_out", type=str, required=True, help="输出 test_small.csv 路径")
    ap.add_argument("--expected_n", type=int, default=1600, help="期望输出行数（默认1600，设为0可不校验）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描 image_dir 子目录")

    # NEW: velodyne 相关参数
    ap.add_argument("--velodyne_dir", type=str, default="", help="velodyne_raw_npy_in_meter 文件夹路径（可选）")
    ap.add_argument("--velodyne_out_dir", type=str, default="", help="将过滤后的 .npy 输出到该目录（可选）")
    ap.add_argument("--velodyne_mode", type=str, default="symlink", choices=["symlink", "copy"],
                    help="输出 velodyne 文件的方式：symlink 或 copy（默认 symlink）")
    ap.add_argument("--require_velodyne", action="store_true",
                    help="若指定，则 CSV 也要进一步按 velodyne 是否存在来过滤（image ∩ velodyne）")

    args = ap.parse_args()

    csv_in = Path(args.csv_in)
    image_dir = Path(args.image_dir)
    csv_out = Path(args.csv_out)

    if not csv_in.exists():
        raise FileNotFoundError(f"csv_in 不存在: {csv_in}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")

    velodyne_dir = Path(args.velodyne_dir) if args.velodyne_dir else None
    velodyne_out_dir = Path(args.velodyne_out_dir) if args.velodyne_out_dir else None
    if velodyne_dir and (not velodyne_dir.exists()):
        raise FileNotFoundError(f"velodyne_dir 不存在: {velodyne_dir}")
    if velodyne_out_dir and (not velodyne_dir):
        raise ValueError("你指定了 --velodyne_out_dir，但没有指定 --velodyne_dir")

    # 1) 读原始CSV
    df = pd.read_csv(csv_in)
    required_cols = [
        "sequence_id", "name", "Scenario", "weather", "severity",
        "Ground_Condition", "Sky_Condition", "Lens_Condition", "Time_of_Day"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少列: {missing_cols}")

    print_stats("Before (original CSV)", df)

    # 2) 扫描图像文件
    img_files = scan_files(image_dir, exts={".png", ".jpg", ".jpeg", ".bmp"}, recursive=args.recursive)
    if len(img_files) == 0:
        raise RuntimeError(f"在 {image_dir} 没有找到图像文件（png/jpg/jpeg/bmp）")

    # 3) 从图像文件生成 (sequence_id, name)
    img_df, bad_img = build_index_df(img_files, parse_image_stem_to_seq_and_name, "image_path")

    if bad_img:
        print("\n[WARN] 以下图像文件名无法解析（将被忽略）：")
        for x in bad_img[:30]:
            print("  ", x)
        if len(bad_img) > 30:
            print(f"  ... 还有 {len(bad_img)-30} 个")

    print(f"\n扫描到可解析且去重后的图像数量: {int(img_df.shape[0])}")

    # 4) 按图像清单过滤 CSV
    merged = img_df.merge(df, on=["sequence_id", "name"], how="left", indicator=True)

    miss = merged[merged["_merge"] != "both"][["sequence_id", "name", "image_path"]]
    if len(miss) > 0:
        print("\n[WARN] 这些图像在 CSV 中找不到对应行（将不会出现在输出CSV里）：")
        for _, r in miss.head(30).iterrows():
            print(f"  - {r['image_path']}  -> ({r['sequence_id']}, {r['name']})")
        if len(miss) > 30:
            print(f"  ... 还有 {len(miss)-30} 个")

    kept = merged[merged["_merge"] == "both"].copy()
    df_new = kept[required_cols].copy().drop_duplicates(subset=["sequence_id", "name"])

    print_stats("After (filtered by current image folder)", df_new)
    print(f"\n当前输出 CSV 行数（仅按 image 过滤后）: {int(df_new.shape[0])}")

    # 5) NEW: velodyne 过滤（可选）
    if velodyne_dir:
        velo_files = scan_files(velodyne_dir, exts={".npy"}, recursive=args.recursive)
        if len(velo_files) == 0:
            raise RuntimeError(f"在 {velodyne_dir} 没有找到 .npy 文件")

        velo_df, bad_velo = build_index_df(velo_files, parse_velodyne_stem_to_seq_and_name, "velodyne_path")

        if bad_velo:
            print("\n[WARN] 以下 velodyne 文件名无法解析（将被忽略）：")
            for x in bad_velo[:30]:
                print("  ", x)
            if len(bad_velo) > 30:
                print(f"  ... 还有 {len(bad_velo)-30} 个")

        print(f"\n扫描到可解析且去重后的 velodyne 数量: {int(velo_df.shape[0])}")

        # 用 df_new（已经按 image 过滤过）去匹配 velodyne
        df_with_velo = df_new.merge(velo_df, on=["sequence_id", "name"], how="left", indicator=True)

        miss_velo = df_with_velo[df_with_velo["_merge"] != "both"][["sequence_id", "name"]]
        hit_velo = df_with_velo[df_with_velo["_merge"] == "both"].copy()

        if len(miss_velo) > 0:
            print("\n[WARN] 这些 (sequence_id, name) 在 velodyne_dir 中找不到对应 .npy：")
            for _, r in miss_velo.head(30).iterrows():
                print(f"  - ({r['sequence_id']}, {r['name']})")
            if len(miss_velo) > 30:
                print(f"  ... 还有 {len(miss_velo)-30} 个")

        print(f"\n与当前 df_new 匹配到的 velodyne 数量: {int(hit_velo.shape[0])} / {int(df_new.shape[0])}")

        # 如果要求 CSV 也必须存在 velodyne，则进一步收缩 df_new
        if args.require_velodyne:
            df_new = hit_velo[required_cols].copy().drop_duplicates(subset=["sequence_id", "name"])
            print(f"\n[require_velodyne] 进一步过滤后 CSV 行数: {int(df_new.shape[0])}")

        # 若指定输出目录，则把匹配到的 velodyne 导出（copy/symlink）
        if velodyne_out_dir:
            # 注意：如果 require_velodyne=False，我们仍然只导出“匹配到的 velodyne”
            # 基于 image-filter 后的 df_new 来匹配 velodyne
            ref_df = df_new.merge(velo_df, on=["sequence_id", "name"], how="inner")
            exported_n = materialize_filtered_files(ref_df, "velodyne_path", velodyne_out_dir, args.velodyne_mode)
            print(f"\n已在 {velodyne_out_dir} 导出 velodyne 文件: {exported_n} 个（mode={args.velodyne_mode}）")

    # 6) 校验输出数量（最终 df_new）
    n_out = int(df_new.shape[0])
    print(f"\n最终输出 CSV 行数: {n_out}")
    if args.expected_n and n_out != args.expected_n:
        raise AssertionError(
            f"输出行数({n_out}) != expected_n({args.expected_n})。\n"
            f"请检查：image目录是否确实为{args.expected_n}张、命名是否一致、或CSV是否缺失对应行。\n"
            f"如果你启用了 --require_velodyne，也会导致行数变少。"
        )

    # 7) 写出 CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(csv_out, index=False)
    print(f"\n已生成: {csv_out}")


if __name__ == "__main__":
    main()

python prune_by_csv.py \
  --csv_small /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small/val_small.csv \
  --base_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small \
  --apply

python prune_by_csv.py \
  --csv_in  /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST_small/test.csv \
  --image_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST_small/image \
  --csv_out  /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST_small/test_small.csv \
  --expected_n 1600 \
  --velodyne_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST_small/velodyne_raw_npy_in_meter
