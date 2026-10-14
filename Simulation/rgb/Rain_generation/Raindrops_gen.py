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
ROOT_DIR  = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset' # Rainfall images root
MASK_PATHS = [
    '/path/to/aurora-root/Dataset/AuroraDepth/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/masks/Large.png',
    '/path/to/aurora-root/Dataset/AuroraDepth/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/masks/medium.png',
    '/path/to/aurora-root/Dataset/AuroraDepth/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/masks/small.png'
]

OUTPUT_DIR= '/path/to/aurora-root/Dataset/AuroraDepth/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/output'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WeatherPathReader:
    def __init__(self, root_dir, split):
        self.root  = root_dir.rstrip('/')
        self.split = split

    def build_image_path(self, name):
        return os.path.join(self.root, 'frame_image', self.split, f"{name}.png")

def random_crop_with_mask(image: Image.Image, crop_size=(375, 1242), mask_size=375,num=3) -> Image.Image:
    """
    Randomly crops an image and applies 2 (possibly overlapping) square transparent masks.

    Parameters:
        image (PIL.Image.Image): Input image (RGBA recommended).
        crop_size (tuple): (height, width) of the crop.
        mask_size (int): Size of square transparent region.

    Returns:
        PIL.Image.Image: Cropped image with transparent square regions (alpha = 0).
    """
    img_width, img_height = image.size
    crop_height, crop_width = crop_size

    if crop_height > img_height or crop_width > img_width:
        raise ValueError("Crop size must be smaller than the image size.")

    # Ensure RGBA format
    image = image.convert("RGBA")

    # Random crop
    top = random.randint(0, img_height - crop_height)
    left = random.randint(0, img_width - crop_width)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))

    # Create alpha channel mask to modify
    r, g, b, a = cropped.split()
    a = a.copy()

    # Draw 3 square transparent regions
    draw = ImageDraw.Draw(a)
    for _ in range(num):
        x1 = random.randint(0, crop_width - mask_size)
        y1 = random.randint(0, crop_height - mask_size)
        draw.rectangle((x1, y1, x1 + mask_size, y1 + mask_size), fill=0)  # alpha = 0 → transparent

    # Merge back channels
    cropped_with_transparency = Image.merge("RGBA", (r, g, b, a))
    return cropped_with_transparency



# ========== 主流程 ==========
if __name__ == '__main__':
    # 读取 CSV 并过滤
    df = pd.read_csv(CSV_PATH)
    df_mod = df[(df.weather=='Rain') &
                (df.severity=='light') &
                (df.get('Lens_Condition')=='raindrop_occluded')]
    total = len(df_mod)
    print(f"🔍 Found {total} entries to process.")

    # 初始化
    reader = WeatherPathReader(ROOT_DIR, SPLIT)
    count   = 0

    for name in tqdm(df_mod['name'], desc="Raindrops on Lens", unit="img"):
        img_path = reader.build_image_path(name)
        if not os.path.isfile(img_path):
            print(f"[skip] image not found: {img_path}")
            continue

        ground = Image.open(img_path).convert("RGBA")
        gw, gh = ground.size

        rain_mask_path = random.choice(MASK_PATHS)
        rain_mask = Image.open(rain_mask_path)

        # rain_mask = Image.open(MASK_PATH)

        rain_mask =  rain_mask.filter(ImageFilter.GaussianBlur(radius=2))

        # Updated
        # result = preprocess_mask_and_apply(ground, rain_mask, blur_radius=5, alpha_ratio=0.7)

        # result = add_raindrop_mask_with_smooth(ground, rain_mask, num=3, mask_size=375)

        rain_mask_crop =random_crop_with_mask(rain_mask, num=3) # Adjust here for how many of the crops 
        rain_mask_crop = rain_mask_crop.filter(ImageFilter.GaussianBlur(radius=3))

        r, g, b, a = rain_mask_crop.split()
        a = a.point(lambda x: int(x * 100 / 255))  # 把最大值255缩放到50，其它按比例缩放
        result = Image.composite(rain_mask_crop, ground, a)

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
