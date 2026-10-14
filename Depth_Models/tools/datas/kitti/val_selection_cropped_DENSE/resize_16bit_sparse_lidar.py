import os
import cv2
import shutil
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


def resize_kitti_depth_folder(
    input_folder,
    output_folder,
    max_examples_print=5,
):
    os.makedirs(output_folder, exist_ok=True)

    # 先把所有图像路径收集好，方便 tqdm 显示总数
    all_files = collect_image_files(input_folder)

    changed_examples = []    # (path, (h,w), (new_h,new_w))
    unchanged_examples = []  # (path, (h,w))
    total = 0
    changed = 0
    unchanged = 0

    for root, fname in tqdm(all_files, desc="Processing images", ncols=80):
        in_path = os.path.join(root, fname)

        # 读取 16-bit 深度图：IMREAD_UNCHANGED 或 -1
        depth = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[WARN] 无法读取图像: {in_path}")
            continue

        # 确保是单通道 & 16-bit
        if depth.dtype != 'uint16':
            print(f"[WARN] 非 uint16 图像，跳过: {in_path}, dtype={depth.dtype}")
            continue
        if len(depth.shape) != 2:
            print(f"[WARN] 非单通道图像，跳过: {in_path}, shape={depth.shape}")
            continue

        h, w = depth.shape[:2]
        total += 1

        # 计算输出路径（保持相对目录结构）
        rel_dir = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        if h == TARGET_HEIGHT and w == TARGET_WIDTH:
            # 尺寸完全一致：直接拷贝文件本身，数值 1:1 不变（甚至不重新编码）
            shutil.copy2(in_path, out_path)
            unchanged += 1
            if len(unchanged_examples) < max_examples_print:
                unchanged_examples.append((in_path, (h, w)))
        else:
            # 需要 resize：用最近邻插值，避免对深度值做平滑
            resized = cv2.resize(
                depth,
                (TARGET_WIDTH, TARGET_HEIGHT),  # 注意：cv2 的顺序是 (width, height)
                interpolation=cv2.INTER_NEAREST
            )
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

    # 打印一些有变化的例子
    print(">>> 尺寸发生变化的例子（最多显示几条）：")
    if not changed_examples:
        print("  无")
    else:
        for p, (h, w), (nh, nw) in changed_examples:
            print(f"  {p}")
            print(f"    {h}x{w} -> {nh}x{nw}")

    print()
    # 打印一些没变化的例子
    print(">>> 尺寸未变化的例子（最多显示几条）：")
    if not unchanged_examples:
        print("  无")
    else:
        for p, (h, w) in unchanged_examples:
            print(f"  {p}")
            print(f"    {h}x{w} (保持不变)")


if __name__ == "__main__":
    # 修改成你自己的输入 / 输出目录
    input_folder = "/path/to/hpc-home/code/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_DENSE/groundtruth_depth_original"
    output_folder = "/path/to/hpc-home/code/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_DENSE/groundtruth_depth"

    resize_kitti_depth_folder(input_folder, output_folder)

