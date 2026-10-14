"""
Compute FID & KID:
- Set A: fixed folder PATH_1 (RGB images)
- Set B: paths selected from CSV by (weather, severity, Time_of_Day),
         resolved to Split_Flattened_Dataset/frame_image/{split}/{name}.png

Install:
    pip install torch-fidelity pillow pandas
# (Optional GPU) install a matching PyTorch build for your CUDA:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Set

# ===================== 配置区（只改这里） =====================
# A 侧：固定目录
PATH_1 = "/path/to/aurora-root/Backup/MUSE_Dataset/muses/frame_camera/train/fog/day"
SEARCH_RECURSIVELY_A = True  # 递归遍历子目录

# B 侧：由 CSV 选择
DATA_ROOT = "/path/to/aurora-root/Dataset/AuroraKITTI/Datasets"
SPLIT = "train"  # 'train' | 'val' | 'test'
CSV_PATH = os.path.join(DATA_ROOT, f"{SPLIT}.csv")
FLAT_ROOT = os.path.join(DATA_ROOT, "Split_Flattened_Dataset")  # 扁平化数据根

# 过滤条件（严格匹配大小写：Day/Night；weather/severity 按 CSV 实际值）
FILTER_WEATHER = "Fog"
FILTER_SEVERITIES = ["light","moderate"]        # 可多选：["light","moderate","heavy"]
FILTER_TIME_OF_DAY = "Day"              # "Day" | "Night" | None(不筛)

# 采样与对齐
MAX_IMAGES_PER_SET = None   # 指定N则每侧最多取N（配合 SAMPLE_POLICY）；None=不限制
SAMPLE_POLICY = "random"    # "random" | "first"
RANDOM_SEED = 42
FORCE_EQUAL_COUNT = True    # 强制两侧等量（取 min(lenA, lenB)），强烈推荐打开

# KID 设置（越大越稳，越慢）
KID_SUBSET_NUM_SUBSETS = 300

# 运行参数
BATCH_SIZE = 1
NUM_WORKERS = 4
FORCE_CUDA = True
# ============================================================

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics


# ---------- utils ----------
def list_images(dir_path: str, recursive: bool) -> List[str]:
    files = []
    if recursive:
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(p)
    files.sort()
    return files


def build_flat_image_path(flat_root: str, split: str, name: str) -> str:
    # 扁平化后的路径规范：{flat_root}/frame_image/{split}/{name}.png
    return os.path.join(flat_root, "frame_image", split, f"{name}.png")


def select_paths_from_csv(csv_path: str, flat_root: str, split: str,
                          weather: str, severities: List[str] = None,
                          time_of_day: str = None) -> List[str]:
    df = pd.read_csv(csv_path)

    if "name" not in df.columns or "weather" not in df.columns or "severity" not in df.columns:
        raise ValueError("CSV 必须包含列：name, weather, severity。")

    # 兼容 Time_of_Day / time_of_day
    tod_col = None
    if "Time_of_Day" in df.columns:
        tod_col = "Time_of_Day"
    elif "time_of_day" in df.columns:
        tod_col = "time_of_day"

    df_sel = df[df["weather"] == weather]
    if severities:
        df_sel = df_sel[df_sel["severity"].isin(severities)]

    if time_of_day is not None:
        if tod_col is None:
            print("[WARN] CSV 中没有 Time_of_Day 列，忽略该条件。")
        else:
            norm = str(time_of_day).strip().capitalize()  # "Day" / "Night"
            df_sel = df_sel[df_sel[tod_col].astype(str).str.capitalize() == norm]

    # 构造真实路径并仅保留存在文件
    paths = []
    for name in df_sel["name"]:
        p = build_flat_image_path(flat_root, split, name)
        if os.path.isfile(p):
            paths.append(p)

    paths.sort()
    return paths


def apply_sampling(paths: List[str], max_n: int, policy: str, seed: int) -> List[str]:
    if max_n is None or max_n <= 0:
        return paths
    n = min(max_n, len(paths))
    if policy == "random":
        random.seed(seed)
        return random.sample(paths, n)
    # default: first
    return paths[:n]


def filter_readable(paths: List[str]) -> List[str]:
    ok = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.convert("RGB")
            ok.append(p)
        except Exception:
            pass
    return ok


def pil_to_chw_uint8_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> torch.uint8 tensor [0,255], CxHxW, with resizable storage."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)  # 可写副本
    t = torch.tensor(arr, dtype=torch.uint8)            # 关键：copy=True
    t = t.permute(2, 0, 1).contiguous()                            # 3xHxW
    return t




class ImagePathsTensorDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            img = pil_to_chw_uint8_tensor(im)  # ← 用 uint8 版本
        return img


# ---------- main ----------
def main():
    # A：固定目录
    if not os.path.isdir(PATH_1):
        raise FileNotFoundError(f"A 目录不存在：{PATH_1}")
    paths_a_all = list_images(PATH_1, SEARCH_RECURSIVELY_A)

    # B：CSV 选择
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV 不存在：{CSV_PATH}")
    paths_b_all = select_paths_from_csv(
        CSV_PATH, FLAT_ROOT, SPLIT,
        weather=FILTER_WEATHER,
        severities=FILTER_SEVERITIES,
        time_of_day=FILTER_TIME_OF_DAY,
    )

    if len(paths_a_all) == 0:
        raise RuntimeError(f"A 目录中未找到图像：{PATH_1}")
    if len(paths_b_all) == 0:
        raise RuntimeError("根据给定过滤条件，B 侧未找到任何图像。")

    # 采样与等量对齐
    max_a = MAX_IMAGES_PER_SET
    max_b = MAX_IMAGES_PER_SET
    if FORCE_EQUAL_COUNT:
        N = min(len(paths_a_all), len(paths_b_all))
        max_a = N if (MAX_IMAGES_PER_SET is None or MAX_IMAGES_PER_SET > N) else MAX_IMAGES_PER_SET
        max_b = max_a

    paths_a = apply_sampling(paths_a_all, max_a, SAMPLE_POLICY, RANDOM_SEED)
    paths_b = apply_sampling(paths_b_all, max_b, SAMPLE_POLICY, RANDOM_SEED)

    # 过滤坏图
    paths_a = filter_readable(paths_a)
    paths_b = filter_readable(paths_b)

    print(f"[Info] A: {PATH_1}  总: {len(paths_a_all)}  用: {len(paths_a)}")
    print(f"[Info] B: from CSV({SPLIT}) weather={FILTER_WEATHER}, severities={FILTER_SEVERITIES}, ToD={FILTER_TIME_OF_DAY}  总: {len(paths_b_all)}  用: {len(paths_b)}")

    if len(paths_a) == 0 or len(paths_b) == 0:
        raise RuntimeError("可用样本为 0，请检查过滤条件/采样开关。")

    # Device
    use_cuda = FORCE_CUDA and torch.cuda.is_available()
    print(f"[Info] 使用CUDA: {use_cuda}")

    # Dataset 方式（稳定）
    ds_a = ImagePathsTensorDataset(paths_a)
    ds_b = ImagePathsTensorDataset(paths_b)

    # KID 子集大小（不能超过两侧样本数）
    kid_subset_size = min(1000, len(paths_a), len(paths_b))
    if kid_subset_size < 5:
        print("[Warn] 样本极少，KID 可能不稳定。")

    metrics = calculate_metrics(
        input1=ds_a,
        input2=ds_b,
        fid=True,
        kid=True,
        verbose=True,
        cuda=use_cuda,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        kid_subset_size=kid_subset_size,
        kid_subset_num_subsets=KID_SUBSET_NUM_SUBSETS,
    )

    kid = metrics.get('kernel_inception_distance_mean', float('nan'))
    kid_std = metrics.get('kernel_inception_distance_std', float('nan'))

    print("\n========== Results ==========")
    print(f"FID : {metrics.get('frechet_inception_distance', float('nan')):.6f}")
    print(f"KID : {kid:.6f}   (x1000 => {kid*1000:.3f})")
    print(f"KID std : {kid_std:.6f}   (x1000 => {kid_std*1000:.3f})")
    print("================================")


if __name__ == "__main__":
    main()
