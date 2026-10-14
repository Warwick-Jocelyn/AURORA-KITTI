# copy_clear_lidar.py
import os
import shutil
import pandas as pd

# ==================== 可配置区域 ====================
BASE_SPLIT_DIR = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"
SRC_DIR = os.path.join(BASE_SPLIT_DIR, "raw_lidar_original")
DST_DIR = os.path.join(BASE_SPLIT_DIR, "raw_lidar_clear")

# CSV 根目录（与你之前的 main.py 一致：/Datasets/{split}.csv）
CSV_ROOT = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets"

# 要处理的 split
SPLITS = ["train", "val", "test"]

# 是否覆盖已存在的目标文件
OVERWRITE = False
# ===================================================

def bin_name_from_csv_name(name: str) -> str:
    """
    由 CSV 的 name 字段还原 .bin 文件名。
    例：
    name: 2011_09_26_drive_0002_sync_image_02_0000000006
    -> 2011_09_26_drive_0002_sync_velodyne_points_0000000006.bin
    """
    parts = name.split("_image_")
    if len(parts) != 2:
        # 非预期格式，直接返回空
        return ""
    prefix = parts[0]             # e.g. 2011_09_26_drive_0002_sync
    suffix = parts[1]             # e.g. 02_0000000006
    frame_id = suffix.split("_")[-1]  # 0000000006
    return f"{prefix}_velodyne_points_{frame_id}.bin"

def process_split(split: str):
    csv_path = os.path.join(CSV_ROOT, f"{split}.csv")
    if not os.path.isfile(csv_path):
        print(f"[Skip {split}] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 只要 Clear 天气，忽略 severity
    df_clear = df[df["weather"] == "Clear"].copy()
    print(f"==> [{split}] Clear rows in CSV: {len(df_clear)}")

    # 去重，避免同一个 bin 被多次复制（因为 name 可能 02/03 两个视角）
    bin_set = set()

    for _, row in df_clear.iterrows():
        name = row.get("name", "")
        if not isinstance(name, str) or name.strip() == "":
            continue
        bin_name = bin_name_from_csv_name(name)
        if not bin_name:
            print(f"[{split}] Bad name format in CSV: {name}")
            continue
        bin_set.add(bin_name)

    print(f"==> [{split}] Unique bins to copy: {len(bin_set)}")

    src_split_dir = os.path.join(SRC_DIR, split)
    dst_split_dir = os.path.join(DST_DIR, split)
    os.makedirs(dst_split_dir, exist_ok=True)

    copied, missing, skipped = 0, 0, 0
    for bin_name in sorted(bin_set):
        src_file = os.path.join(src_split_dir, bin_name)
        dst_file = os.path.join(dst_split_dir, bin_name)

        if not os.path.isfile(src_file):
            print(f"[Missing] {src_file}")
            missing += 1
            continue

        if os.path.exists(dst_file) and not OVERWRITE:
            # 已存在且不覆盖
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"[{split}] Done. Copied: {copied}, Skipped(existing): {skipped}, Missing: {missing}")

def main():
    print("Start copying Clear-weather LiDAR bins...")
    print(f"From: {SRC_DIR}")
    print(f"To  : {DST_DIR}")
    for split in SPLITS:
        process_split(split)
    print("All splits finished.")

if __name__ == "__main__":
    main()
