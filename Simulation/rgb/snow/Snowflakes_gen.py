#!/usr/bin/env python3
import os
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter

# —— 配置区：在此修改 ——
PDF_ROOT = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets'  # CSV 所在根目录
SPLIT     = 'train'                                 # 可选: 'train', 'val', 'test'
CSV_PATH  = os.path.join(PDF_ROOT, f"{SPLIT}.csv")
ROOT_DIR  = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset'
MASK_PATH = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/medium.png'
OUTPUT_DIR= '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset'
NUM_MASKS = 2     # 每张图随机挖 NUM_MASKS 个孔
BLUR_SIZE = 35    # 雪花 mask 高斯模糊核大小
ALPHA_MAX = 180   # 雪花透明度最大值

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WeatherPathReader:
    def __init__(self, root_dir, split):
        self.root  = root_dir.rstrip('/')
        self.split = split

    def build_image_path(self, name):
        return os.path.join(self.root, 'frame_image', self.split, f"{name}.png")


def create_base_snow(mask_path, blur_size=BLUR_SIZE, alpha_max=ALPHA_MAX):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Snow mask not found: {mask_path}")
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    h, w = blurred.shape
    snow = np.zeros((h, w, 4), dtype=np.uint8)
    snow[..., :3] = 255
    alpha = (blurred.astype(np.float32) / 255.0 * alpha_max).astype(np.uint8)
    snow[..., 3] = alpha
    return Image.fromarray(snow)


def random_crop_with_snow_mask(overlay: Image.Image,
                               crop_size: tuple,
                               mask_size: int,
                               num: int = NUM_MASKS) -> Image.Image:
    img_w, img_h = overlay.size
    crop_h, crop_w = crop_size
    top  = random.randint(0, img_h - crop_h)
    left = random.randint(0, img_w - crop_w)
    cropped = overlay.crop((left, top, left + crop_w, top + crop_h))
    r, g, b, a = cropped.split()
    draw = ImageDraw.Draw(a)
    for _ in range(num):
        x1 = random.randint(0, crop_w - mask_size)
        y1 = random.randint(0, crop_h - mask_size)
        draw.rectangle((x1, y1, x1 + mask_size, y1 + mask_size), fill=0)
    return Image.merge("RGBA", (r, g, b, a))


def colorize_snow_mask(mask_rgba: Image.Image) -> Image.Image:
    arr = np.array(mask_rgba)
    alpha = arr[..., 3] > 0
    h, w = alpha.shape
    gray = np.random.randint(80, 230, size=(h, w), dtype=np.uint8)
    gray = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=3))
    gray_vals = np.array(gray)
    arr[..., 0][alpha] = (gray_vals * 0.98)[alpha]
    arr[..., 1][alpha] = (gray_vals * 0.90)[alpha]
    arr[..., 2][alpha] = (gray_vals * 0.85)[alpha]
    return Image.fromarray(arr, mode='RGBA')

# ========== 主流程 ==========
if __name__ == '__main__':
    # 读取 CSV 并过滤
    df = pd.read_csv(CSV_PATH)
    df_mod = df[(df.weather=='Snow') &
                (df.severity=='moderate') &
                (df.get('Lens_Condition')=='snowflake_occluded')]
    total = len(df_mod)
    print(f"🔍 Found {total} entries to process.")

    # 初始化
    reader = WeatherPathReader(ROOT_DIR, SPLIT)
    base    = create_base_snow(MASK_PATH)
    bw, bh  = base.size
    count   = 0

    for name in tqdm(df_mod['name'], desc="Snowflake overlays", unit="img"):
        img_path = reader.build_image_path(name)
        if not os.path.isfile(img_path):
            print(f"[skip] image not found: {img_path}")
            continue

        ground = Image.open(img_path).convert("RGBA")
        gw, gh = ground.size
        snow_big = base.resize((bw*2, bh*2), resample=Image.BILINEAR)
        mask_crop = random_crop_with_snow_mask(
            snow_big, crop_size=(gh, gw), mask_size=min(gw, gh, bw)
        )
        mask_blur = mask_crop.filter(ImageFilter.GaussianBlur(radius=3))
        mask_col  = colorize_snow_mask(mask_blur)
        result    = Image.alpha_composite(ground, mask_col)

        rel       = os.path.relpath(img_path, start=ROOT_DIR)
        out_path  = os.path.join(OUTPUT_DIR, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result.convert("RGB").save(out_path)
        count   += 1

    print(f"🎉 Done: total={total}, generated={count}")
    if count == total:
        print("✅ All done, no misses!")
    else:
        print("⚠️ Mismatch: some entries skipped.")
