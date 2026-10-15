import cv2
import numpy as np
import os

def convert_16bit_to_8bit(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        if filename.lower().endswith(('.png', '.tif', '.tiff')):
            filepath = os.path.join(src_folder, filename)
            img16 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            if img16 is None:
                print(f"无法读取图像: {filepath}")
                continue

            # 线性归一化到0-255
            img16 = img16.astype(np.float32)
            # img_norm = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)
            img_norm = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)
            img8 = img_norm.astype(np.uint8)

            # 保存
            save_path = os.path.join(dst_folder, filename)
            cv2.imwrite(save_path, img8)
            print(f"保存 8-bit 图像: {save_path}")

# 用法示例
src = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth16bit"
dst = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth8bit"
convert_16bit_to_8bit(src, dst)
