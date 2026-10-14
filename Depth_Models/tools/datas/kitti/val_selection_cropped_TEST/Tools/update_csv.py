#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_image_stem_to_seq_and_name(stem: str):
    """
    支持两种常见格式：
    1) 文件名格式（你的例子）:
       {sequence_id}_image_{frame}_image_02
       -> name = {sequence_id}_image_02_{frame}
    2) 已经是 CSV name 格式：
       {sequence_id}_image_02_{frame}
    frame 允许非10位数字，最终会 zfill(10) 来对齐 CSV（通常是10位）
    """
    # A: seq_image_0000000005_image_02  (frame在中间，camera在末尾)
    m = re.match(r"(.+)_image_(\d+)_((?:image)_\d{2})$", stem)
    if m:
        seq, frame, cam = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    # B: seq_image_02_0000000005 (已经是CSV name风格)
    m = re.match(r"(.+)_((?:image)_\d{2})_(\d+)$", stem)
    if m:
        seq, cam, frame = m.group(1), m.group(2), m.group(3)
        frame = frame.zfill(10)
        name = f"{seq}_{cam}_{frame}"
        return seq, name

    raise ValueError(f"无法解析图像名: {stem}")


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

    # weather
    total_w, rows_w = stat_counts(df["weather"], ["Clear", "Fog", "Rain", "Snow"])
    print(f"[weather] total={total_w}")
    for cat, c, r in rows_w:
        print(f"  - {cat:5s}: {c:4d}  ({r*100:6.2f}%)")

    # Time_of_Day
    total_t, rows_t = stat_counts(df["Time_of_Day"], ["Day", "Night"])
    print(f"[Time_of_Day] total={total_t}")
    for cat, c, r in rows_t:
        print(f"  - {cat:5s}: {c:4d}  ({r*100:6.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", type=str, required=True, help="原始 test.csv 路径")
    ap.add_argument("--image_dir", type=str, required=True, help="当前 image 文件夹路径")
    ap.add_argument("--csv_out", type=str, required=True, help="输出 test_small.csv 路径")
    ap.add_argument("--expected_n", type=int, default=1600, help="期望输出行数（默认1600，设为0可不校验）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描 image_dir 子目录")
    args = ap.parse_args()

    csv_in = Path(args.csv_in)
    image_dir = Path(args.image_dir)
    csv_out = Path(args.csv_out)

    if not csv_in.exists():
        raise FileNotFoundError(f"csv_in 不存在: {csv_in}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")

    # 1) 读原始CSV
    df = pd.read_csv(csv_in)
    required_cols = [
        "sequence_id", "name", "Scenario", "weather", "severity",
        "Ground_Condition", "Sky_Condition", "Lens_Condition", "Time_of_Day"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少列: {missing_cols}")

    # 统计删减前分布
    print_stats("Before (original CSV)", df)

    # 2) 扫描图像文件
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if args.recursive:
        files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]

    files = sorted(files, key=lambda p: p.name)
    if len(files) == 0:
        raise RuntimeError(f"在 {image_dir} 没有找到图像文件（png/jpg/jpeg/bmp）")

    # 3) 从图像文件生成 (sequence_id, name)
    img_rows = []
    bad_files = []
    for p in files:
        stem = p.stem  # 不带后缀
        try:
            seq, name = parse_image_stem_to_seq_and_name(stem)
            img_rows.append({"sequence_id": seq, "name": name, "image_path": str(p)})
        except Exception:
            bad_files.append(str(p))

    if bad_files:
        print("\n[WARN] 以下文件名无法解析（将被忽略）：")
        for x in bad_files[:30]:
            print("  ", x)
        if len(bad_files) > 30:
            print(f"  ... 还有 {len(bad_files)-30} 个")

    img_df = pd.DataFrame(img_rows).drop_duplicates(subset=["sequence_id", "name"])
    n_images = int(img_df.shape[0])
    print(f"\n扫描到可解析且去重后的图像数量: {n_images}")

    # 4) 按图像清单过滤 CSV（按行删）
    # 为了保证输出顺序跟图像文件一致，用 img_df 左连接 df
    merged = img_df.merge(df, on=["sequence_id", "name"], how="left", indicator=True)

    # 找出 image 中存在但 CSV 找不到对应行的情况
    miss = merged[merged["_merge"] != "both"][["sequence_id", "name", "image_path"]]
    if len(miss) > 0:
        print("\n[WARN] 这些图像在 CSV 中找不到对应行（将不会出现在输出CSV里）：")
        for _, r in miss.head(30).iterrows():
            print(f"  - {r['image_path']}  -> ({r['sequence_id']}, {r['name']})")
        if len(miss) > 30:
            print(f"  ... 还有 {len(miss)-30} 个")

    # 只保留匹配成功的行，并保持列顺序与原CSV一致
    kept = merged[merged["_merge"] == "both"].copy()
    df_new = kept[required_cols].copy()

    # 去重（保险）
    df_new = df_new.drop_duplicates(subset=["sequence_id", "name"])

    # 统计删减后分布
    print_stats("After (filtered by current image folder)", df_new)

    # 5) 校验输出数量
    n_out = int(df_new.shape[0])
    print(f"\n输出 CSV 行数: {n_out}")
    if args.expected_n and n_out != args.expected_n:
        raise AssertionError(
            f"输出行数({n_out}) != expected_n({args.expected_n})。\n"
            f"请检查：image目录是否确实为{args.expected_n}张、命名是否一致、或CSV是否缺失对应行。"
        )

    # 6) 写出
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(csv_out, index=False)
    print(f"\n已生成: {csv_out}")


if __name__ == "__main__":
    main()


# python update_csv.py \
#   --csv_in /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small/val.csv \
#   --image_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small/image \
#   --csv_out /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL_small//val_small.csv \
#   --expected_n 1600
