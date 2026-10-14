import os
import re
import random
from glob import glob
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm


# =========================
# 路径读取与数据筛选
# =========================
class WeatherPathReader:
    def __init__(self, csv_path, root_dir, split):
        self.df    = pd.read_csv(csv_path)
        self.root  = root_dir.rstrip('/')  # 用于后面 relativize
        self.split = split
        self.modality_map = {
            'image':     ('frame_image',      '.png'),
            '8bitdepth': ('8bit_clean_depth', '.png'),
        }

    def build_image_path(self, name):
        folder, ext = self.modality_map['image']
        return os.path.join(self.root, folder, self.split, f"{name}{ext}")

    def build_depth_path(self, name):
        # 如果你不需要深度，这个函数不会被用到；保留以兼容
        seq, tail = name.split('_sync_')
        depth_tail = tail.replace('image', '8bitdepth', 1)
        filename = f"{seq}_sync_{depth_tail}.png"
        folder, ext = self.modality_map['8bitdepth']
        return os.path.join(self.root, folder, self.split, filename)


# =========================
# 掩码处理（PIL, RGBA）
# =========================
def random_crop_mask(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """crop_size=(H, W)"""
    img_w, img_h = image.size
    crop_h, crop_w = crop_size
    if crop_h > img_h or crop_w > img_w:
        raise ValueError(f"Crop size {crop_size} must be <= mask image size {(img_h, img_w)}.")
    left = random.randint(0, img_w - crop_w)
    top  = random.randint(0, img_h - crop_h)
    return image.crop((left, top, left + crop_w, top + crop_h))

def create_snow_mask(mask_path: str) -> Image.Image:
    """将黑底白雪的灰度 mask 转 RGBA：白色区域作为 alpha。"""
    gray = Image.open(mask_path).convert("L")
    snow = Image.new("RGBA", gray.size, (255, 255, 255, 0))
    snow.putalpha(gray)
    return snow

def prepare_mask_layer(mask_path: str, crop_size, blur_radius) -> Image.Image:
    """读取→两倍放大→随机裁剪到 crop_size→高斯模糊，返回 RGBA"""
    layer = create_snow_mask(mask_path)
    layer = layer.resize((layer.width * 2, layer.height * 2), resample=Image.LANCZOS)
    layer = random_crop_mask(layer, crop_size)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return layer


# =========================
# 掩码文件收集
# =========================
def list_mask_files(mask_dir: str, exts=(".png", ".jpg", ".jpeg")) -> List[str]:
    files = []
    for e in exts:
        files.extend(glob(os.path.join(mask_dir, f"*{e}")))
    # 按 01, 02, ... 的数字顺序排序（忽略前导零）
    def key_fn(p):
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)$", base)
        return int(m.group(1)) if m else base
    return sorted(files, key=key_fn)


# =========================
# 合成：根据 severity 决定使用 small 或 medium（含 small 叠加）
# =========================
def composite_by_severity(
    ground_path: str,
    severity: str,
    small_masks: List[str],
    medium_masks: List[str],
    blur_radius_small: int = 0,
    blur_radius_medium: int = 0
) -> Optional[np.ndarray]:
    """
    - light   -> 使用 small 掩码：仅 small
    - moderate/medium -> 使用 medium 掩码：medium + 随机 small 叠加
    - 其他 severity：返回 None（上层跳过或你可改成默认用 medium）
    返回 OpenCV BGR 图像
    """
    sev = (severity or "").strip().lower()
    if sev not in {"light", "moderate", "medium"}:
        # 如果希望 heavy 也走 medium，可把下一行改为： sev = "medium"
        return None

    ground = Image.open(ground_path).convert("RGBA")
    h, w = ground.size[1], ground.size[0]
    crop_size = (h, w)

    if sev == "light":
        small_mask_path = random.choice(small_masks)
        small_layer = prepare_mask_layer(small_mask_path, crop_size, blur_radius_small)
        out = Image.alpha_composite(ground, small_layer)
    else:
        # medium / moderate：medium + small 叠加
        medium_mask_path = random.choice(medium_masks)
        medium_layer = prepare_mask_layer(medium_mask_path, crop_size, blur_radius_medium)
        if small_masks:  # 再叠加一层 small
            small_on_medium = prepare_mask_layer(random.choice(small_masks), crop_size, blur_radius_small)
            medium_layer = Image.alpha_composite(medium_layer, small_on_medium)
        out = Image.alpha_composite(ground, medium_layer)

    # 转 OpenCV BGR
    out_bgr = cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out_bgr


# =========================
# 主流程（批量生成）
# =========================
if __name__ == '__main__':
    # —— 配置区 —— 
    SPLIT      = 'train'    # train / val / test
    CSV_PATH   = f"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/{SPLIT}.csv"
    ROOT_DIR   = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    # 掩码目录
    SMALL_MASK_DIR  = "/path/to/aurora-root/Dataset/SRRS-2021/SmallSnow"
    MEDIUM_MASK_DIR = "/path/to/aurora-root/Dataset/SRRS-2021/MidSnow"

    # —— 新的输出根目录：保持原文件名与相对目录结构 —— 
    OUTPUT_DIR = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    # 读取 CSV：仅保留天气为 Snow 的条目，其余可按需放开
    df = pd.read_csv(CSV_PATH)
    df_snow = df[(df.weather == 'Snow')]

    total = len(df_snow)
    print(f"🔍 CSV 中 weather=Snow 的条目数：{total}")

    # 收集掩码
    small_masks  = list_mask_files(SMALL_MASK_DIR)
    medium_masks = list_mask_files(MEDIUM_MASK_DIR)
    if not small_masks:
        raise FileNotFoundError(f"[ERROR] Small masks not found in: {SMALL_MASK_DIR}")
    if not medium_masks:
        raise FileNotFoundError(f"[ERROR] Medium masks not found in: {MEDIUM_MASK_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wpr = WeatherPathReader(CSV_PATH, ROOT_DIR, SPLIT)

    generated = 0
    skipped   = 0

    # 遍历每条记录，按 severity 合成并保存到 OUTPUT_DIR（文件名保持不变）
    for name, severity in tqdm(zip(df_snow['name'], df_snow['severity']),
                               total=len(df_snow), desc="Compositing by severity", unit="img"):
        img_path = wpr.build_image_path(name)
        if not os.path.isfile(img_path):
            print(f"[跳过] 找不到底图：{img_path}")
            skipped += 1
            continue

        result_bgr = composite_by_severity(
            ground_path=img_path,
            severity=severity,
            small_masks=small_masks,
            medium_masks=medium_masks,
            blur_radius_small=2,
            blur_radius_medium=3
        )
        if result_bgr is None:
            # print(f"[跳过] 未定义的 severity='{severity}'（可在 composite_by_severity 中定义映射）")
            skipped += 1
            continue

        # 保持相对路径和文件名不变
        rel = os.path.relpath(img_path, start=ROOT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, result_bgr)
        generated += 1

    # 结果校验
    print(f"🎉 生成完毕：目标（Snow）={total}, 实际生成={generated}")
    if generated == total:
        print("✅ 数量一致，未遗漏！")
    else:
        print("⚠️ 数量不符，请检查跳过项（路径、severity 映射或缺失文件）。")
