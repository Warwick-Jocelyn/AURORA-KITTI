import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 目标尺寸（height, width）
TARGET_HEIGHT = 352
TARGET_WIDTH = 1216

# 支持的图像后缀
EXTS = {'.png', '.jpg', '.jpeg', '.tiff'}


def collect_image_files(input_folder):
    """收集所有需要处理的图像文件路径"""
    files = []
    for root, _, fnames in os.walk(input_folder):
        for fname in fnames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in EXTS:
                files.append((root, fname))
    return files


def _channels_are_equal(depth_bgr: np.ndarray) -> bool:
    """
    检查三通道是否一致（做一个稀疏采样检查，避免全量比较太慢/太占内存）
    OpenCV 读进来是 BGR 顺序。
    """
    h, w = depth_bgr.shape[:2]
    # 取一个大概 50x50 左右的网格采样
    step_h = max(1, h // 50)
    step_w = max(1, w // 50)
    sample = depth_bgr[::step_h, ::step_w, :]
    return np.array_equal(sample[:, :, 0], sample[:, :, 1]) and np.array_equal(sample[:, :, 0], sample[:, :, 2])


def resize_kitti_depth_folder_24bit_3ch(
    input_folder,
    output_folder,
    max_examples_print=5,
):
    os.makedirs(output_folder, exist_ok=True)

    all_files = collect_image_files(input_folder)

    changed_examples = []    # (path, (h,w), (new_h,new_w))
    unchanged_examples = []  # (path, (h,w))
    total = 0
    changed = 0
    unchanged = 0

    for root, fname in tqdm(all_files, desc="Processing images", ncols=80):
        in_path = os.path.join(root, fname)

        # 读取 24-bit 三通道图：用 IMREAD_UNCHANGED 保持原始格式
        depth = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[WARN] 无法读取图像: {in_path}")
            continue

        # 期望：uint8 + 三通道（24-bit RGB/BGR）
        if depth.dtype != np.uint8:
            print(f"[WARN] 非 uint8 图像，跳过: {in_path}, dtype={depth.dtype}")
            continue
        if depth.ndim != 3 or depth.shape[2] != 3:
            print(f"[WARN] 非三通道图像，跳过: {in_path}, shape={depth.shape}")
            continue

        h, w = depth.shape[:2]
        total += 1

        # 可选：检查三通道是否一致（仅采样检查，不一致也继续处理，但会提醒）
        if not _channels_are_equal(depth):
            print(f"[WARN] 三通道数值似乎不一致(采样检测): {in_path}")

        # 计算输出路径（保持相对目录结构）
        rel_dir = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        if h == TARGET_HEIGHT and w == TARGET_WIDTH:
            # 尺寸一致：直接拷贝文件本身，避免重编码
            shutil.copy2(in_path, out_path)
            unchanged += 1
            if len(unchanged_examples) < max_examples_print:
                unchanged_examples.append((in_path, (h, w)))
        else:
            # 需要 resize：最近邻，避免深度值被平滑
            resized = cv2.resize(
                depth,
                (TARGET_WIDTH, TARGET_HEIGHT),  # cv2: (width, height)
                interpolation=cv2.INTER_NEAREST
            )

            # 如果你希望“强制”三通道完全一致（防止上游有微小差异），可打开下面三行：
            # ch0 = resized[:, :, 0]
            # resized[:, :, 1] = ch0
            # resized[:, :, 2] = ch0

            cv2.imwrite(out_path, resized)
            changed += 1
            if len(changed_examples) < max_examples_print:
                new_h, new_w = resized.shape[:2]
                changed_examples.append((in_path, (h, w), (new_h, new_w)))

    # 打印统计信息
    print("\n====== 处理完成 ======")
    print(f"总图像数: {total}")
    print(f"尺寸改变的图像数: {changed}")
    print(f"尺寸未改变的图像数: {unchanged}")
    print()

    print(">>> 尺寸发生变化的例子（最多显示几条）：")
    if not changed_examples:
        print("  无")
    else:
        for p, (h, w), (nh, nw) in changed_examples:
            print(f"  {p}")
            print(f"    {h}x{w} -> {nh}x{nw}")

    print()
    print(">>> 尺寸未变化的例子（最多显示几条）：")
    if not unchanged_examples:
        print("  无")
    else:
        for p, (h, w) in unchanged_examples:
            print(f"  {p}")
            print(f"    {h}x{w} (保持不变)")


if __name__ == "__main__":
    # 修改成你自己的输入 / 输出目录
    input_folder = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/disp_clean_original"
    output_folder = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/disp_clean"

    resize_kitti_depth_folder_24bit_3ch(input_folder, output_folder)
