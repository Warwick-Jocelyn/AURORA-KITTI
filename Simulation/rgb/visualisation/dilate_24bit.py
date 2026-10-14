import cv2
import numpy as np
import os
import argparse

def dilate_sparse_lidar(input_path, output_path, radius=2, bg_threshold=250):
    """
    对白色背景的稀疏点云投影图做“半径为 radius 的膨胀”，
    让彩色点/线变粗。
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[错误] 无法读取图像: {input_path}")
        return

    # 1. 认为“接近白色”的是背景，其余是点云
    #    bg_threshold=250 给一点余量（防 JPEG/插值带来的 254/253）
    mask = np.any(img < bg_threshold, axis=2)  # True 表示前景点

    if not np.any(mask):
        print(f"[警告] {input_path} 中未找到前景点，直接拷贝原图")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return

    # 2. 构造前景图：只保留点云颜色，背景置黑
    foreground = img.copy()
    foreground[~mask] = 0

    # 3. 构造半径为 radius 的圆形结构元素
    kernel_size = radius * 2 + 1       # radius=2 → 5x5
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # 4. 对前景做膨胀，让点/线变粗
    dilated_foreground = cv2.dilate(foreground, kernel, iterations=1)

    # 5. 把膨胀后的前景贴回白色背景
    result = np.full_like(img, 255)               # 全白
    dilated_mask = np.any(dilated_foreground > 0, axis=2)
    result[dilated_mask] = dilated_foreground[dilated_mask]

    # 6. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(
        f"[完成] {os.path.basename(input_path)} -> "
        f"{os.path.basename(output_path)} (radius={radius}, kernel={kernel_size}x{kernel_size})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="输入图像文件夹（白色背景点云投影）")
    parser.add_argument("--out-dir", required=True, help="输出图像文件夹")
    parser.add_argument(
        "--dilate",
        type=int,
        default=2,
        help="膨胀半径（像素），radius=2 对应 5x5 圆形核",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exts = [".png", ".jpg", ".jpeg", ".bmp"]

    for name in os.listdir(args.in_dir):
        if not any(name.lower().endswith(e) for e in exts):
            continue
        in_path = os.path.join(args.in_dir, name)
        out_path = os.path.join(args.out_dir, name)
        dilate_sparse_lidar(in_path, out_path, radius=args.dilate)


if __name__ == "__main__":
    main()

# python dilate_24bit.py \
#     --in-dir /path/to/aurora-root/Code/Depth_visualisation/white_input \
#     --out-dir /path/to/aurora-root/Code/Depth_visualisation/updated_output \
#     --dilate 5
