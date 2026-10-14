"""
Simple FID between two RGB image folders (stable sampling + Tensor Dataset).

Install:
    pip install torch-fidelity pillow
# (Optional GPU) Install a matching PyTorch build for your CUDA, e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

import os
import random
import numpy as np
from typing import List, Set, Union

# ============ 只改这里：两个路径 ============
PATH_1 = "/path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/RGB_validate/Samples/Aurorakitti-fog-day"
PATH_2 = "/path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/RGB_validate/Samples/muses-fog-day"
# =========================================

# 运行选项
BATCH_SIZE = 64
NUM_WORKERS = 4
FORCE_CUDA = True
SEARCH_RECURSIVELY = True  # 递归遍历子目录

# 采样开关（任选其一；不需要就保持 None）
MAX_IMAGES_PER_SET = None   # 例如 200；为 None 则不采样，也不会用 Dataset 包装
SAMPLE_POLICY = "first"     # "first" 或 "random"
RANDOM_SEED = 42            # "random" 时生效

# 允许的图片后缀
IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics


def list_images(dir_path: str, recursive: bool = True) -> List[str]:
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


def apply_sampling(paths: List[str]) -> List[str]:
    if MAX_IMAGES_PER_SET is None or MAX_IMAGES_PER_SET <= 0:
        return paths
    n = min(MAX_IMAGES_PER_SET, len(paths))
    if SAMPLE_POLICY == "random":
        random.seed(RANDOM_SEED)
        return random.sample(paths, n)
    # default: "first"
    return paths[:n]


def filter_readable(paths: List[str]) -> List[str]:
    """过滤掉打不开的图片，避免中途炸。"""
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
    """PIL -> torch.uint8 tensor, CxHxW, [0,255]"""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)  # HxWx3, [0,255]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3xHxW
    return t

class ImagePathsTensorDataset(Dataset):
    """Dataset 返回 torch.uint8 tensor (CxHxW, [0,255])，满足 torch-fidelity 要求。"""
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            img = pil_to_chw_uint8_tensor(im)  # 注意这里使用 uint8
        return img


def main():
    # 基本检查
    if not os.path.isdir(PATH_1):
        raise FileNotFoundError(f"目录不存在：{PATH_1}")
    if not os.path.isdir(PATH_2):
        raise FileNotFoundError(f"目录不存在：{PATH_2}")

    # 收集
    paths1_all = list_images(PATH_1, SEARCH_RECURSIVELY)
    paths2_all = list_images(PATH_2, SEARCH_RECURSIVELY)
    if len(paths1_all) == 0:
        raise RuntimeError(f"未在目录中找到图像：{PATH_1}")
    if len(paths2_all) == 0:
        raise RuntimeError(f"未在目录中找到图像：{PATH_2}")
    
    # 保证两个目录样本数一致
    N = min(len(paths1_all), len(paths2_all))
    MAX_IMAGES_PER_SET = N
    SAMPLE_POLICY = "random"
    RANDOM_SEED = 42
    
    # 采样
    paths1_sel = apply_sampling(paths1_all)
    paths2_sel = apply_sampling(paths2_all)

    # 过滤坏图
    paths1_sel = filter_readable(paths1_sel)
    paths2_sel = filter_readable(paths2_sel)

    print(f"[Info] 目录A: {PATH_1}  总图像: {len(paths1_all)}  用于计算: {len(paths1_sel)}")
    print(f"[Info] 目录B: {PATH_2}  总图像: {len(paths2_all)}  用于计算: {len(paths2_sel)}")

    # CUDA
    use_cuda = FORCE_CUDA and torch.cuda.is_available()
    print(f"[Info] 使用CUDA: {use_cuda}")

    if len(paths1_sel) == 0 or len(paths2_sel) == 0:
        raise RuntimeError("可用样本为 0，请检查目录或采样开关。")

    # Dataset 输入
    if MAX_IMAGES_PER_SET is None or MAX_IMAGES_PER_SET <= 0:
        input1: Union[str, Dataset] = PATH_1
        input2: Union[str, Dataset] = PATH_2
    else:
        input1 = ImagePathsTensorDataset(paths1_sel)
        input2 = ImagePathsTensorDataset(paths2_sel)

    # --- 只计算 FID ---
    metrics = calculate_metrics(
        input1=input1,
        input2=input2,
        fid=True,
        cuda=use_cuda,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        samples_find_deep=SEARCH_RECURSIVELY if isinstance(input1, str) and isinstance(input2, str) else False,
        verbose=True
    )

    print("\n========== Results ==========")
    print(f"FID : {metrics.get('frechet_inception_distance', float('nan')):.6f}")
    print("================================")


if __name__ == "__main__":
    main()
