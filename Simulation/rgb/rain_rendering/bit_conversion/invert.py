import os
from PIL import Image
import numpy as np

def invert_and_darken_16bit_grayscale(input_folder, output_folder, darken_factor=0.3):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)

            # if img.mode != 'I;16':
            #     print(f"跳过非16位图像: {filename}")
            #     continue

            # 转换为numpy数组
            gray_array = np.array(img, dtype=np.uint16)

            # 反色处理
            inverted_array = 30000 - gray_array

            # 调暗处理（乘以系数），并截断在0~65535范围内
            darkened_array = np.clip(inverted_array * darken_factor, 0, 30000).astype(np.uint16)

            # 保存图像
            output_img = Image.fromarray(darkened_array, mode='I;16')
            output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.png')
            output_img.save(output_path)

            print(f"已反色并调暗: {filename} → {output_path}")

# 示例用法
input_dir = '/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2_ori/depth16bit'
output_dir = '/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2_ori/depth'
invert_and_darken_16bit_grayscale(input_dir, output_dir, darken_factor=0.5)
