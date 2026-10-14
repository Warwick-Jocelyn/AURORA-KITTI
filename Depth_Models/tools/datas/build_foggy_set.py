"""
Build AURORA_Foggy_Set from KITTI dataset CSVs.

Filters 'Fog' weather entries from train/val/test CSVs and copies:
  - RGB images  -> AURORA_Foggy_Set/RGB/{Train,Val,Test}/{Day,Night}/
  - Depth GT    -> AURORA_Foggy_Set/Depth_annotation/{Train,Val,Test}/{Day,Night}/
  - Fog CSVs    -> AURORA_Foggy_Set/{train,val,test}_fog.csv

Output filename convention (matches val_selection_cropped naming):
  {seq_id}_image_{frame_id}_{camera}.png
  e.g. 2011_09_26_drive_0001_sync_image_0000000005_image_02.png

Sources:
  Train RGB   : raw/{date}/{seq}/{camera}/data/{frame}.png
  Train Depth : data_depth_annotated/train/{seq}/proj_depth/groundtruth/{camera}/{frame}.png
  Val RGB     : val_selection_cropped_VAL/image/{seq}_image_{frame}_{camera}.png
  Val Depth   : val_selection_cropped_VAL/groundtruth_depth/{seq}_groundtruth_depth_{frame}_{camera}.png
  Test RGB    : val_selection_cropped_TEST/image/{seq}_image_{frame}_{camera}.png
  Test Depth  : val_selection_cropped_TEST/groundtruth_depth/{seq}_groundtruth_depth_{frame}_{camera}.png
"""

import csv
import re
import shutil
from pathlib import Path

BASE = Path("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas")
KITTI = BASE / "kitti"
OUTPUT = BASE / "AURORA_Foggy_Set"

SPLIT_CONFIGS = [
    {
        "name": "train",
        "csv": KITTI / "train.csv",
        "out_folder": "Train",
        "get_rgb": lambda seq, camera, frame, date: (
            KITTI / "raw" / date / seq / camera / "data" / f"{frame}.png"
        ),
        "get_depth": lambda seq, camera, frame, date: (
            KITTI / "data_depth_annotated" / "train" / seq /
            "proj_depth" / "groundtruth" / camera / f"{frame}.png"
        ),
    },
    {
        "name": "val",
        "csv": KITTI / "val.csv",
        "out_folder": "Val",
        "get_rgb": lambda seq, camera, frame, date: (
            KITTI / "val_selection_cropped_VAL" / "image" /
            f"{seq}_image_{frame}_{camera}.png"
        ),
        "get_depth": lambda seq, camera, frame, date: (
            KITTI / "val_selection_cropped_VAL" / "groundtruth_depth" /
            f"{seq}_groundtruth_depth_{frame}_{camera}.png"
        ),
    },
    {
        "name": "test",
        "csv": KITTI / "test.csv",
        "out_folder": "Test",
        "get_rgb": lambda seq, camera, frame, date: (
            KITTI / "val_selection_cropped_TEST" / "image" /
            f"{seq}_image_{frame}_{camera}.png"
        ),
        "get_depth": lambda seq, camera, frame, date: (
            KITTI / "val_selection_cropped_TEST" / "groundtruth_depth" /
            f"{seq}_groundtruth_depth_{frame}_{camera}.png"
        ),
    },
]

NAME_RE = re.compile(r'^(.+)_(image_\d+)_(\d+)$')
DATE_RE = re.compile(r'^(\d{4}_\d{2}_\d{2})_')


def parse_name(name):
    """'2011_09_26_drive_0001_sync_image_02_0000000005'
    -> (seq_id='2011_09_26_drive_0001_sync', camera='image_02', frame='0000000005')
    """
    m = NAME_RE.match(name.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def get_date(seq_id):
    """'2011_09_26_drive_0001_sync' -> '2011_09_26'"""
    m = DATE_RE.match(seq_id)
    return m.group(1) if m else None


def process_split(config):
    out_folder = config["out_folder"]
    fog_rows = []
    stats = {
        "Day":   {"rgb_ok": 0, "rgb_miss": 0, "depth_ok": 0, "depth_miss": 0},
        "Night": {"rgb_ok": 0, "rgb_miss": 0, "depth_ok": 0, "depth_miss": 0},
    }

    # Pre-create Day / Night subdirectories
    for tod in ("Day", "Night"):
        (OUTPUT / "RGB" / out_folder / tod).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "Depth_annotation" / out_folder / tod).mkdir(parents=True, exist_ok=True)

    with open(config["csv"], newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            weather = row["weather"].strip()
            if weather != "Fog":
                continue

            parsed = parse_name(row["name"])
            if parsed is None:
                print(f"  [WARN] Cannot parse name: {row['name']}")
                continue

            seq_id, camera, frame_id = parsed
            date = get_date(seq_id)
            if date is None:
                print(f"  [WARN] Cannot extract date from: {seq_id}")
                continue

            # Determine Day / Night sub-folder
            tod = row["Time_of_Day"].strip()
            if tod not in ("Day", "Night"):
                print(f"  [WARN] Unexpected Time_of_Day '{tod}' for {row['name']}, skipping")
                continue

            rgb_out   = OUTPUT / "RGB" / out_folder / tod
            depth_out = OUTPUT / "Depth_annotation" / out_folder / tod

            # Output filename (consistent format for all splits)
            out_name = f"{seq_id}_image_{frame_id}_{camera}.png"

            # ---- RGB ----
            rgb_src = config["get_rgb"](seq_id, camera, frame_id, date)
            rgb_dst = rgb_out / out_name
            if rgb_src.exists():
                shutil.copy2(rgb_src, rgb_dst)
                stats[tod]["rgb_ok"] += 1
            else:
                stats[tod]["rgb_miss"] += 1
                print(f"  [MISSING RGB]   {rgb_src}")

            # ---- Depth GT ----
            depth_src = config["get_depth"](seq_id, camera, frame_id, date)
            depth_dst = depth_out / out_name
            if depth_src.exists():
                shutil.copy2(depth_src, depth_dst)
                stats[tod]["depth_ok"] += 1
            else:
                stats[tod]["depth_miss"] += 1
                print(f"  [MISSING DEPTH] {depth_src}")

            fog_rows.append(row)

    # Save filtered CSV
    fog_csv_path = OUTPUT / f"{config['name']}_fog.csv"
    with open(fog_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fog_rows)

    return fog_rows, stats


def main():
    # Remove old flat output if it exists (from previous run without Day/Night)
    for split_folder in ("Train", "Val", "Test"):
        for modality in ("RGB", "Depth_annotation"):
            old_dir = OUTPUT / modality / split_folder
            if old_dir.is_dir():
                # Delete any .png files directly in split_folder (not in Day/Night sub-dirs)
                for f in old_dir.glob("*.png"):
                    f.unlink()

    print(f"Output directory: {OUTPUT}\n")
    total_fog = 0

    for config in SPLIT_CONFIGS:
        split = config["name"]
        print(f"Processing {split}.csv ...")
        fog_rows, stats = process_split(config)
        n = len(fog_rows)
        total_fog += n
        print(f"  Fog entries : {n}")
        for tod in ("Day", "Night"):
            s = stats[tod]
            print(f"  [{tod}] RGB copied: {s['rgb_ok']}  missing: {s['rgb_miss']}  |"
                  f"  Depth copied: {s['depth_ok']}  missing: {s['depth_miss']}")
        print(f"  CSV saved  -> {OUTPUT}/{split}_fog.csv\n")

    print(f"Finished. Total fog entries across all splits: {total_fog}")
    print(f"\nDataset layout:")
    print(f"  {OUTPUT}/RGB/{{Train,Val,Test}}/{{Day,Night}}/")
    print(f"  {OUTPUT}/Depth_annotation/{{Train,Val,Test}}/{{Day,Night}}/")
    print(f"  {OUTPUT}/train_fog.csv   val_fog.csv   test_fog.csv")


if __name__ == "__main__":
    main()
