# """
# Compute FID & KID (Kernel Inception Distance):
# - Set A: fixed folder PATH_1 (RGB images)
# - Set B: paths selected from CSV by filters (weather, severity, Time_of_Day, Scenario, Lens_Condition),
#          resolved to Split_Flattened_Dataset/frame_image/{split}/{name}.png
# """

import os
import random
import numpy as np
import pandas as pd
from typing import List, Set
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics

# ===================== 配置区（只改这里） =====================
# A 侧：固定目录
PATH_1 = "/path/to/aurora-root/Backup/MUSE_Dataset/muses/frame_camera/train/rain/night"
# PATH_1 = "/path/to/aurora-root/Backup/MUSE_Dataset/muses/frame_camera/train/clear/day"
SEARCH_RECURSIVELY_A = True  # 递归遍历子目录

# B 侧：由 CSV 选择
DATA_ROOT = "/path/to/aurora-root/Dataset/AuroraKITTI/Datasets"
SPLIT = "test"  # 'train' | 'val' | 'test'
CSV_PATH = os.path.join(DATA_ROOT, f"{SPLIT}.csv")
FLAT_ROOT = os.path.join(DATA_ROOT, "Split_Flattened_Dataset")  # 扁平化数据根

# -------- 新增/完善的过滤条件（大小写/空格不敏感）--------
FILTER_WEATHER       = "Rain"                 # 如：'Rain'/'Snow'/'Clear'；None 表示不筛
FILTER_SEVERITIES    =  'light'         # 多选，如 ['light','moderate','heavy'] ["None"]；None 表示不筛
FILTER_TIME_OF_DAY   = "Night"                   # 'Day'|'Night'；None 表示不筛
FILTER_SCENARIOS     = None           # Scenario 多选 ["City"]  ；None 不筛
FILTER_LENS_COND     = None  # Lens_Condition 多选["raindrop_occluded" snowflake_occluded ] ；None 不筛
# ------------------------------------------------------------

# 采样与对齐
MAX_IMAGES_PER_SET = None   # 指定N则每侧最多取N（配合 SAMPLE_POLICY）；None=不限制
SAMPLE_POLICY = "random"    # "random" | "first"
RANDOM_SEED = 42
FORCE_EQUAL_COUNT = True    # 强制两侧等量（min(lenA,lenB)）

# KID 设置
KID_SUBSET_NUM_SUBSETS = 300

# 运行参数
BATCH_SIZE = 1
NUM_WORKERS = 0
FORCE_CUDA = True

# A/B 侧增强设置
# 可用：Brightness, Contrast, Sharpness, Color
# A_ENHANCERS = [("Brightness", 1.4), ("Sharpness", 1.2)]  # A侧增加亮度 20%
# B_ENHANCERS  = [("Brightness", 0.8)]   # B侧增加锐度 2倍
A_ENHANCERS = [("Brightness", 1.0), ("Contrast", 0.8)]  # A侧增加亮度 20%
B_ENHANCERS  = [("Brightness", 1.0),("Contrast", 1.0)]   # B侧增加锐度 2倍

# ============================================================

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# ----------------- 工具函数 -----------------
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
    return os.path.join(flat_root, "frame_image", split, f"{name}.png")

def _norm_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x.strip().casefold()]  # 字符串 -> 单元素列表
    return [str(v).strip().casefold() for v in x]  # 列表 -> 遍历

# def _norm_list(x):
#     if x is None:
#         return None
#     return [str(v).strip().casefold() for v in x]


def select_paths_from_csv(csv_path: str, flat_root: str, split: str,
                          weather: str = None,
                          severities: List[str] = None,
                          time_of_day: str = None,
                          scenarios: List[str] = None,
                          lens_conditions: List[str] = None) -> List[str]:
    df = pd.read_csv(csv_path)
    for col in ["name","weather","severity","Time_of_Day","time_of_day","Scenario","Lens_Condition"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "name" not in df.columns:
        raise ValueError("CSV 必须包含列：name。")
    tod_col = "Time_of_Day" if "Time_of_Day" in df.columns else ("time_of_day" if "time_of_day" in df.columns else None)
    df_norm = df.copy()
    for col in ["weather","severity","Scenario","Lens_Condition"]:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].astype(str).str.strip().str.casefold()
    if tod_col:
        df_norm[tod_col] = df_norm[tod_col].astype(str).str.strip().str.casefold()
    if weather is not None and "weather" in df_norm.columns:
        df_norm = df_norm[df_norm["weather"] == str(weather).strip().casefold()]
    severities = _norm_list(severities)
    if severities is not None and "severity" in df_norm.columns:
        df_norm = df_norm[df_norm["severity"].isin(severities)]
    if time_of_day is not None and tod_col:
        df_norm = df_norm[df_norm[tod_col] == str(time_of_day).strip().casefold()]
    scenarios = _norm_list(scenarios)
    if scenarios is not None and "Scenario" in df_norm.columns:
        df_norm = df_norm[df_norm["Scenario"].isin(scenarios)]
    lens_conditions = _norm_list(lens_conditions)
    if lens_conditions is not None and "Lens_Condition" in df_norm.columns:
        df_norm = df_norm[df_norm["Lens_Condition"].isin(lens_conditions)]
    paths = []
    for name in df_norm["name"]:
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
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    t = torch.tensor(arr, dtype=torch.uint8)
    t = t.permute(2,0,1).contiguous()
    return t


def apply_enhance(img: Image.Image, enhancers: list) -> Image.Image:
    if enhancers is None:
        return img
    for enhancer in enhancers:
        name, factor = enhancer
        if name.lower() == "brightness":
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif name.lower() == "contrast":
            img = ImageEnhance.Contrast(img).enhance(factor)
        elif name.lower() == "sharpness":
            img = ImageEnhance.Sharpness(img).enhance(factor)
        elif name.lower() == "color":
            img = ImageEnhance.Color(img).enhance(factor)
    return img

class ImagePathsTensorDataset(Dataset):
    def __init__(self, paths: List[str], enhancer: tuple = None):
        self.paths = paths
        self.enhancer = enhancer
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = apply_enhance(im, self.enhancer)
            return pil_to_chw_uint8_tensor(im)


# ----------------- main -----------------
def main():
    if not os.path.isdir(PATH_1):
        raise FileNotFoundError(f"A 目录不存在：{PATH_1}")
    paths_a_all = list_images(PATH_1, SEARCH_RECURSIVELY_A)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV 不存在：{CSV_PATH}")
    paths_b_all = select_paths_from_csv(
        CSV_PATH, FLAT_ROOT, SPLIT,
        weather=FILTER_WEATHER,
        severities=FILTER_SEVERITIES,
        time_of_day=FILTER_TIME_OF_DAY,
        scenarios=FILTER_SCENARIOS,
        lens_conditions=FILTER_LENS_COND,
    )

    if len(paths_a_all) == 0:
        raise RuntimeError(f"A 目录中未找到图像：{PATH_1}")
    if len(paths_b_all) == 0:
        raise RuntimeError("根据过滤条件，B 侧未找到任何图像。")

    max_a = MAX_IMAGES_PER_SET
    max_b = MAX_IMAGES_PER_SET
    if FORCE_EQUAL_COUNT:
        N = min(len(paths_a_all), len(paths_b_all))
        max_a = N if (MAX_IMAGES_PER_SET is None or MAX_IMAGES_PER_SET > N) else MAX_IMAGES_PER_SET
        max_b = max_a

    paths_a = apply_sampling(paths_a_all, max_a, SAMPLE_POLICY, RANDOM_SEED)
    paths_b = apply_sampling(paths_b_all, max_b, SAMPLE_POLICY, RANDOM_SEED)

    paths_a = filter_readable(paths_a)
    paths_b = filter_readable(paths_b)

    print(f"[Info] A: {PATH_1}  总: {len(paths_a_all)}  用: {len(paths_a)}")
    print(f"[Info] B: from CSV({SPLIT}) "
          f"weather={FILTER_WEATHER}, severities={FILTER_SEVERITIES}, ToD={FILTER_TIME_OF_DAY}, "
          f"Scenario={FILTER_SCENARIOS}, Lens_Condition={FILTER_LENS_COND}  "
          f"总: {len(paths_b_all)}  用: {len(paths_b)}")

    if len(paths_a) == 0 or len(paths_b) == 0:
        raise RuntimeError("可用样本为 0，请检查过滤条件/采样开关。")

    use_cuda = FORCE_CUDA and torch.cuda.is_available()
    print(f"[Info] 使用CUDA: {use_cuda}")

    ds_a = ImagePathsTensorDataset(paths_a, enhancer=A_ENHANCERS)
    ds_b = ImagePathsTensorDataset(paths_b, enhancer=B_ENHANCERS)

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
