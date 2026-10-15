

import os
from PIL import Image
import numpy as np

def convert_24bit_to_16bit_grayscale(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)

            # 转换为灰度（确保是8-bit灰度图）
            gray_img = img.convert('L')  # 'L' 是 8-bit 灰度

            # 转换为16-bit数据（简单方式：*257）
            gray_array = np.array(gray_img, dtype=np.uint8)
            gray_16bit = (gray_array.astype(np.uint16)) * 257  # 映射到 0-65535

            # 保存为 16-bit PNG
            out_img = Image.fromarray(gray_16bit, mode='I;16')
            output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '_16bit.png')
            out_img.save(output_path)

            print(f"转换完成: {filename} → {output_path}")

# 示例用法
input_dir = '/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth0'
output_dir = '/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth'
convert_24bit_to_16bit_grayscale(input_dir, output_dir)
