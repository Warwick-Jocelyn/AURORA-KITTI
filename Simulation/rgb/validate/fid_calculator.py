"""
FID & KID for explicit image file paths (PNG/JPG/WEBP/TIFF/BMP).
Install:
    pip install torch-fidelity pillow
# (Optional GPU) Install a matching PyTorch build for your CUDA:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

import os
import glob
from typing import List, Set, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics

# ============ 在这里填入“具体图片路径” ============
# 1) 直接写绝对/相对路径；2) 支持通配符（glob）；3) 也可写 ".txt"（每行一个路径）
IMAGES_1 = [
    "/path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/RGB_validate/Samples/Aurora-fog-day.png",
    # "/path/to/A/0002.png",
    # 也可以写 "A_dir/*.png",
    # 或 "/path/list_A.txt"
]
IMAGES_2 = [
    "/path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/RGB_validate/Samples/muse-fog-day.png",
    # "/path/to/B/0002.png",
    # "B_dir/**/*.png",
    # "/path/list_B.txt"
]
# ===============================================

# 运行选项
BATCH_SIZE = 64
NUM_WORKERS = 4
FORCE_CUDA = True

# 当两边数量不等时是否强制对齐（截断到较小数量，保持确定性）
FORCE_EQUAL_COUNT = True

# 少样本时如何处理 KID
# 如果 min(nA,nB) < KID_MIN_REQUIRED 且 ONLY_FID_IF_TOO_FEW=True，则关闭 KID 仅计算 FID
KID_MIN_REQUIRED = 5
ONLY_FID_IF_TOO_FEW = True

# 正常（样本数足够）时的 KID 子集设置（越小越快、方差略大）
KID_SUBSET_NUM_SUBSETS = 300

# 允许的图片后缀
IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def expand_inputs(specs: List[str]) -> List[str]:
    """
    把用户给的路径规格展开成实际文件列表：
    - 直接文件路径：保留
    - 通配符：glob 展开
    - .txt：逐行读取路径
    """
    out: List[str] = []
    for s in specs:
        if s.lower().endswith(".txt"):
            if not os.path.isfile(s):
                raise FileNotFoundError(f"列表文件不存在: {s}")
            with open(s, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        out.append(p)
        elif any(sym in s for sym in ["*", "?", "["]):
            out.extend(glob.glob(s, recursive=True))
        else:
            out.append(s)

    # 只保留存在且后缀允许的文件
    keep: List[str] = []
    for p in out:
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
            keep.append(os.path.abspath(p))
    keep = sorted(list(dict.fromkeys(keep)))  # 去重并排序，稳定
    return keep


def filter_readable(paths: List[str]) -> List[str]:
    """过滤掉打不开的图片，避免中途报错。"""
    ok = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.convert("RGB")
            ok.append(p)
        except Exception:
            # 可根据需要打印警告
            pass
    return ok


def pil_to_chw_float_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> torch.float32 tensor in [0,1], CxHxW."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # HxWx3
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()         # 3xHxW
    return t


class ImagePathsTensorDataset(Dataset):
    """Dataset: 返回 torch.Tensor(CxHxW, float32, [0,1])，满足 torch-fidelity 要求。"""
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            img = pil_to_chw_float_tensor(im)
        return img


def main():
    paths1 = expand_inputs(IMAGES_1)
    paths2 = expand_inputs(IMAGES_2)

    if len(paths1) == 0:
        raise RuntimeError("IMAGES_1 为空或未找到匹配图片。")
    if len(paths2) == 0:
        raise RuntimeError("IMAGES_2 为空或未找到匹配图片。")

    # 过滤坏图
    paths1 = filter_readable(paths1)
    paths2 = filter_readable(paths2)

    if len(paths1) == 0 or len(paths2) == 0:
        raise RuntimeError("可用样本为 0，请检查图片路径或格式。")

    # 数量对齐（可选）
    if FORCE_EQUAL_COUNT and len(paths1) != len(paths2):
        n = min(len(paths1), len(paths2))
        paths1 = paths1[:n]
        paths2 = paths2[:n]

    print(f"[Info] 集合A: {len(paths1)} 张")
    print(f"[Info] 集合B: {len(paths2)} 张")

    # Dataset 方式（能精确控制样本集合）
    input1: Union[str, Dataset] = ImagePathsTensorDataset(paths1)
    input2: Union[str, Dataset] = ImagePathsTensorDataset(paths2)
    nA, nB = len(paths1), len(paths2)

    # CUDA
    use_cuda = FORCE_CUDA and torch.cuda.is_available()
    print(f"[Info] 使用CUDA: {use_cuda}")

    # KID 参数自适应
    kid_enabled = True
    kid_subset_size = min(1000, nA, nB)
    kid_num_subsets = KID_SUBSET_NUM_SUBSETS

    if min(nA, nB) < KID_MIN_REQUIRED:
        if ONLY_FID_IF_TOO_FEW:
            kid_enabled = False
            print(f"[Warn] 样本过少（min={min(nA,nB)}），已关闭 KID，仅计算 FID。")
        else:
            kid_subset_size = min(nA, nB)  # 尽量小但有效
            kid_num_subsets = 1
            print(f"[Warn] 样本过少（min={min(nA,nB)}），KID 可能不稳定；已将 kid_subset_size={kid_subset_size}, num_subsets=1。")

    metrics = calculate_metrics(
        input1=input1,
        input2=input2,
        fid=True,
        kid=kid_enabled,
        verbose=True,
        cuda=use_cuda,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        kid_subset_size=kid_subset_size,
        kid_subset_num_subsets=kid_num_subsets,
        samples_find_deep=False,  # 我们直接传 Dataset，不需要递归找样本
    )

    print("\n========== Results ==========")
    print(f"FID : {metrics.get('frechet_inception_distance', float('nan')):.6f}")
    if kid_enabled:
        print(f"KID : {metrics.get('kernel_inception_distance_mean', float('nan')):.6f}")
        print(f"KID std : {metrics.get('kernel_inception_distance_std', float('nan')):.6f}")
    else:
        print("KID : disabled (too few samples)")
    print("================================")


if __name__ == "__main__":
    main()
