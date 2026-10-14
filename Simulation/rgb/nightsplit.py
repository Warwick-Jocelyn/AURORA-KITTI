import os
import shutil
import pandas as pd
from tqdm import tqdm

class PathReader:
    """按 split 构建 GT 彩色图路径（与您之前逻辑一致）"""
    def __init__(self, root_dir: str, split: str):
        self.root = root_dir.rstrip('/')
        self.split = split
        self.folder = 'frame_image'
        self.ext = '.png'

    def build_image_path(self, name: str) -> str:
        return os.path.join(self.root, self.folder, self.split, f"{name}{self.ext}")

def pick_tod_column(df: pd.DataFrame) -> str:
    for c in ['Time_of_Day', 'time_of_day', 'timeOfDay', 'TimeOfDay']:
        if c in df.columns:
            return c
    raise KeyError("CSV 中未找到 Time_of_Day 列（可尝试修改列名列表）。")

if __name__ == "__main__":
    # ------------ 配置区（按需修改） ------------
    SPLITS   = ["train", "val", "test"]
    CSV_DIR  = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets"  # 各 split 的 csv 所在目录
    ROOT_DIR = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    # 输出根：将 frame_image 改为 frame_image-nightsplit
    # 目标路径示例：ROOT_DIR/frame_image-nightsplit/val/<name>.png
    OUT_FOLDER_NAME = "frame_image-nightsplit"
    # ----------------------------------------

    total, copied = 0, 0
    for split in SPLITS:
        csv_path = os.path.join(CSV_DIR, f"{split}.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        tod_col = pick_tod_column(df)
        # 只保留 Night
        df_night = df[df[tod_col].astype(str).str.lower() == "night"]

        reader = PathReader(ROOT_DIR, split)

        names = df_night['name'].tolist() if 'name' in df_night.columns else []
        total += len(names)

        for name in tqdm(names, desc=f"[{split}] copying Night images", unit="img"):
            src = reader.build_image_path(name)
            # 目标：ROOT_DIR/frame_image-nightsplit/<split>/<same filename>.png
            rel = os.path.relpath(src, start=os.path.join(ROOT_DIR, "frame_image"))
            dst = os.path.join(ROOT_DIR, OUT_FOLDER_NAME, rel)

            if not os.path.isfile(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"Done. Night entries={total}, copied={copied}.")
