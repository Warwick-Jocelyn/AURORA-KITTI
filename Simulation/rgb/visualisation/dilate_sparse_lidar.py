import cv2
import numpy as np
import os
import argparse

def dilate_sparse_lidar(input_path, output_path, dilate_size=3):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[错误] 无法读取图像: {input_path}")
        return

    # 1. 找到非白色点
    sparse_mask = np.any(img < 250, axis=-1)

    if not np.any(sparse_mask):
        print(f"[警告] 图像中未找到非白色点，直接拷贝原图: {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return

    # 2. 只保留前景（点云），背景置黑
    foreground = img.copy()
    foreground[~sparse_mask] = 0

    # 3. 膨胀前景
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    dilated_foreground = cv2.dilate(foreground, kernel, iterations=1)

    # 4. 将背景恢复为白色
    result = np.full_like(img, 255)
    dilated_mask = np.any(dilated_foreground > 0, axis=-1)
    result[dilated_mask] = dilated_foreground[dilated_mask]

    # 5. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[完成] {input_path} -> {output_path}（dilate_size={dilate_size}）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="输入图像文件夹（点云投影，白色背景）")
    parser.add_argument("--out-dir", required=True, help="输出结果文件夹")
    parser.add_argument("--dilate", type=int, default=3, help="膨胀核大小（奇数，例如 3/5/7）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exts = [".png", ".jpg", ".jpeg", ".bmp"]

    for name in os.listdir(args.in_dir):
        if not any(name.lower().endswith(e) for e in exts):
            continue

        in_path = os.path.join(args.in_dir, name)
        out_path = os.path.join(args.out_dir, name)

        dilate_sparse_lidar(in_path, out_path, dilate_size=args.dilate)

if __name__ == "__main__":
    main()

# python dilate_sparse_lidar.py \
#     --in-dir /path/to/aurora-root/Code/Depth_visualisation/Output \
#     --out-dir /path/to/aurora-root/Code/Depth_visualisation/dilate_output \
#     --dilate 3
