# import os
# import cv2

# def invert_depth_images(input_dir: str, output_dir: str):
#     # 支持的图像后缀
#     valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
#     os.makedirs(output_dir, exist_ok=True)

#     for fname in os.listdir(input_dir):
#         name, ext = os.path.splitext(fname)
#         if ext.lower() not in valid_exts:
#             continue  # 跳过非图像文件

#         in_path = os.path.join(input_dir, fname)
#         # 以原样（保留单通道）读取
#         img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"无法读取: {in_path}")
#             continue

#         # 检查是否为 8 位单通道图
#         if img.dtype != 'uint8':
#             print(f"警告: {fname} 不是 8-bit 图像，已跳过")
#             continue
#         if len(img.shape) != 2:
#             print(f"警告: {fname} 不是单通道图像，已跳过")
#             continue

#         # 反色操作
#         inverted = 255 - img

#         out_path = os.path.join(output_dir, fname)
#         cv2.imwrite(out_path, inverted)
#         print(f"已保存反色图: {out_path}")

# if __name__ == "__main__":
#     # 在此处定义输入和输出路径
#     input_dir = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2_ori/depth8bit"
#     output_dir = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2_ori/depth8bit_invert"

#     invert_depth_images(input_dir, output_dir)

import os
import cv2
import numpy as np

def invert_depth_images(input_dir: str, output_dir: str, base: int = 200):
    """
    对 8-bit 深度图做“反色”操作，以 base - 像素值 的方式进行，并对溢出及负值进行饱和处理。
    base: 反转基准值（默认为 200）。
    """
    # 支持的图像后缀
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in valid_exts:
            continue  # 跳过非图像文件

        in_path = os.path.join(input_dir, fname)
        # 以原样（保留单通道）读取
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取: {in_path}")
            continue

        # 检查是否为 8 位单通道图
        if img.dtype != np.uint8:
            print(f"警告: {fname} 不是 8-bit 图像，已跳过")
            continue
        if img.ndim != 2:
            print(f"警告: {fname} 不是单通道图像，已跳过")
            continue

        # 使用 cv2.subtract 进行饱和减法，避免负值环绕
        inverted = cv2.subtract(base, img)
        # 或者使用 numpy 手动实现并裁剪：
        # inverted = np.clip(base - img.astype(np.int16), 0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, inverted)
        print(f"已保存反色图: {out_path}")

if __name__ == "__main__":
    # 在此处定义输入和输出路径，以及反转基准值
    input_dir = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth8bit"
    output_dir = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth"
    base_value = 255  # 设置反色基准线

    invert_depth_images(input_dir, output_dir, base=base_value)
