# generate_missing_report.py
import os
import re
import argparse
import pandas as pd

# ==================== 默认配置（可改） ====================
NOHUP_PATH_DEFAULT = "/path/to/aurora-root/nohup.out"  # 你的 nohup.out 路径（按需修改）
OUTPUT_CSV_DEFAULT = "/path/to/aurora-root/missing.csv"

TRAIN_CSV_PATH = "/path/to/aurora-root/PhD_Project_2025/lidar_compound_noise_models_original/raw_lidar/train.csv"
# =========================================================

MISS_PREFIX = "[Skip] Missing lidar bin:"
BIN_RE = re.compile(
    r".*\[Skip\]\s+Missing lidar bin:\s+(?P<path>.+?/Split_Flattened_Dataset/raw_lidar/.+?/(?P<bin>[^/]+\.bin))"
)

def parse_missing_bins(nohup_path: str):
    """从 nohup.out 解析缺失的 bin 路径、split、bin 文件名。"""
    missing = []
    if not os.path.isfile(nohup_path):
        print(f"[Error] nohup file not found: {nohup_path}")
        return missing

    with open(nohup_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if MISS_PREFIX not in line:
                continue
            m = BIN_RE.match(line.strip())
            if not m:
                # 兜底：直接寻找路径片段
                path = line.split(MISS_PREFIX, 1)[-1].strip()
                split = "unknown"
                if "/raw_lidar/train/" in path:
                    split = "train"
                elif "/raw_lidar/val/" in path:
                    split = "val"
                elif "/raw_lidar/test/" in path:
                    split = "test"
                bin_name = os.path.basename(path)
                missing.append((split, path, bin_name))
                continue
            path = m.group("path")
            bin_name = m.group("bin")
            # split 从路径中粗略提取
            if "/raw_lidar/train/" in path:
                split = "train"
            elif "/raw_lidar/val/" in path:
                split = "val"
            elif "/raw_lidar/test/" in path:
                split = "test"
            else:
                split = "unknown"
            missing.append((split, path, bin_name))

    # 去重
    missing = sorted(list(set(missing)))
    print(f"Parsed {len(missing)} unique missing bins from nohup.")
    return missing

def bin_name_to_keys(bin_name: str):
    """
    由 bin 文件名构造在 CSV 里匹配的关键信息。
    例如：2011_09_26_drive_0087_sync_velodyne_points_0000000511.bin
    -> sequence_prefix: 2011_09_26_drive_0087_sync
       frame_id: 0000000511
       两个可能的 name：
         2011_09_26_drive_0087_sync_image_02_0000000511
         2011_09_26_drive_0087_sync_image_03_0000000511
    """
    # 去后缀
    stem = bin_name[:-4] if bin_name.endswith(".bin") else bin_name
    # 末尾是帧号
    if "_velodyne_points_" not in stem:
        return None
    prefix, frame_id = stem.split("_velodyne_points_")
    seq_prefix = prefix  # e.g. 2011_09_26_drive_0087_sync
    name02 = f"{seq_prefix}_image_02_{frame_id}"
    name03 = f"{seq_prefix}_image_03_{frame_id}"
    return {
        "sequence_prefix": seq_prefix,
        "frame_id": frame_id,
        "candidate_names": [name02, name03],
    }

def load_train_csv(train_csv_path: str):
    if not os.path.isfile(train_csv_path):
        print(f"[Error] train.csv not found: {train_csv_path}")
        return None
    df = pd.read_csv(train_csv_path)
    # 统一列名集合（有的表可能没有 id）
    expected_cols = [
        "id", "sequence_id", "name", "Scenario", "weather", "severity",
        "Ground_Condition", "Sky_Condition", "Lens_Condition", "Time_of_Day"
    ]
    # 确保列存在（缺失的列会在最终输出时填空）
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate missing.csv by cross-referencing nohup.out and train.csv.")
    parser.add_argument("--nohup", default=NOHUP_PATH_DEFAULT, help="Path to nohup.out")
    parser.add_argument("--out", default=OUTPUT_CSV_DEFAULT, help="Output CSV path")
    parser.add_argument("--train_csv", default=TRAIN_CSV_PATH, help="Path to train.csv for cross-reference")
    args = parser.parse_args()

    # 1) 解析 missing bins
    missing = parse_missing_bins(args.nohup)
    if not missing:
        print("No missing bins found. Exit.")
        return

    # 2) 载入 train.csv
    df_train = load_train_csv(args.train_csv)
    if df_train is None:
        return

    rows_out = []
    for split, bin_path, bin_name in missing:
        keys = bin_name_to_keys(bin_name)
        if keys is None:
            # 无法解析，就只输出 bin 基本信息
            rows_out.append({
                "split": split,
                "bin_path": bin_path,
                "bin_name": bin_name,
                "id": pd.NA,
                "sequence_id": pd.NA,
                "name": pd.NA,
                "Scenario": pd.NA,
                "weather": pd.NA,
                "severity": pd.NA,
                "Ground_Condition": pd.NA,
                "Sky_Condition": pd.NA,
                "Lens_Condition": pd.NA,
                "Time_of_Day": pd.NA,
            })
            continue

        cand_names = keys["candidate_names"]  # 两个相机视角
        # 只在 train.csv 里查（你的需求即“refer to train.csv”）
        # 由于同一帧在 02/03 都会有一条记录，这里把两条都保留写入
        hits = df_train[df_train["name"].isin(cand_names)]

        if hits.empty:
            # 未匹配到，也写一条占位，便于回查
            rows_out.append({
                "split": split,
                "bin_path": bin_path,
                "bin_name": bin_name,
                "id": pd.NA if "id" not in df_train.columns else pd.NA,
                "sequence_id": pd.NA,
                "name": pd.NA,
                "Scenario": pd.NA,
                "weather": pd.NA,
                "severity": pd.NA,
                "Ground_Condition": pd.NA,
                "Sky_Condition": pd.NA,
                "Lens_Condition": pd.NA,
                "Time_of_Day": pd.NA,
            })
        else:
            # 将命中的多行逐条加入
            for _, r in hits.iterrows():
                rows_out.append({
                    "split": split,
                    "bin_path": bin_path,
                    "bin_name": bin_name,
                    "id": r.get("id", pd.NA),
                    "sequence_id": r.get("sequence_id", pd.NA),
                    "name": r.get("name", pd.NA),
                    "Scenario": r.get("Scenario", pd.NA),
                    "weather": r.get("weather", pd.NA),
                    "severity": r.get("severity", pd.NA),
                    "Ground_Condition": r.get("Ground_Condition", pd.NA),
                    "Sky_Condition": r.get("Sky_Condition", pd.NA),
                    "Lens_Condition": r.get("Lens_Condition", pd.NA),
                    "Time_of_Day": r.get("Time_of_Day", pd.NA),
                })

    # 3) 输出 CSV
    out_cols = [
        "split", "bin_path", "bin_name",
        "id", "sequence_id", "name",
        "Scenario", "weather", "severity",
        "Ground_Condition", "Sky_Condition", "Lens_Condition", "Time_of_Day",
    ]
    df_out = pd.DataFrame(rows_out, columns=out_cols).drop_duplicates()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"✅ Saved report to: {args.out}")
    print(f"Rows: {len(df_out)}")

if __name__ == "__main__":
    main()


# python generate_missing_report.py \
#   --nohup /path/to/aurora-root/Dataset/AuroraDepth/Codes_Lidar_Generation/Projection/nohup.out \
#   --out   /path/to/aurora-root/Dataset/AuroraDepth/Codes_Lidar_Generation/Projection/missing.csv
