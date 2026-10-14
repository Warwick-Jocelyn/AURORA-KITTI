import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 目标尺寸（height, width）
TARGET_HEIGHT = 375
TARGET_WIDTH = 1242

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


def resize_and_convert_disparity_folder(
    input_folder,
    output_folder,
    max_examples_print=5,
    max_stats_print=10,
):
    """
    对 KITTI dense disparity map 进行处理：
    1) 读取 24bit 三通道 (uint8, BGR)，每通道数值相同
    2) 打印每个通道的最大值（若干示例）
    3) 如果尺寸 != (TARGET_HEIGHT, TARGET_WIDTH) 则使用 INTER_LINEAR resize
    4) 将三通道取平均后 * 256，得到 16bit 单通道 disparity
    5) 保存到新的 output_folder 中
    6) 重新读取新图，验证 dtype / shape / max value
    """
    os.makedirs(output_folder, exist_ok=True)

    all_files = collect_image_files(input_folder)

    changed_examples = []    # (path, (h,w), (new_h,new_w))
    unchanged_examples = []  # (path, (h,w))
    total = 0
    changed = 0
    unchanged = 0

    # 存放原始 & 转换后的一些统计示例
    orig_stats_examples = []   # (path, dtype, shape, [max_c0, max_c1, max_c2])
    new_stats_examples = []    # (path, dtype, shape, max_val)

    for root, fname in tqdm(all_files, desc="Processing disparity maps", ncols=80):
        in_path = os.path.join(root, fname)

        # 读取 24-bit 三通道 disparity（IMREAD_UNCHANGED 保持原始 bit depth）
        disp = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            print(f"[WARN] 无法读取图像: {in_path}")
            continue

        # 要求为 3 通道 uint8（典型 24bit BGR）
        if len(disp.shape) != 3 or disp.shape[2] != 3:
            print(f"[WARN] 不是 3 通道图像，跳过: {in_path}, shape={disp.shape}")
            continue
        if disp.dtype != np.uint8:
            print(f"[WARN] 非 uint8 图像（非 24bit 三通道），跳过: {in_path}, dtype={disp.dtype}")
            continue

        h, w = disp.shape[:2]
        total += 1

        # 统计原始通道最大值（仅打印前 max_stats_print 个示例）
        if len(orig_stats_examples) < max_stats_print:
            ch_max = [int(disp[..., c].max()) for c in range(3)]
            orig_stats_examples.append((in_path, disp.dtype, disp.shape, ch_max))

        # 计算输出路径（保持相对目录结构）
        rel_dir = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        # Step 1: 尺寸处理（仅针对尺寸统计 changed / unchanged）
        if h == TARGET_HEIGHT and w == TARGET_WIDTH:
            # 尺寸未变（但后续仍要做 3->1 通道 + 8bit->16bit 的转换）
            disp_resized = disp
            unchanged += 1
            if len(unchanged_examples) < max_examples_print:
                unchanged_examples.append((in_path, (h, w)))
        else:
            # 需要 resize：对 dense disparity 使用双线性插值
            disp_resized = cv2.resize(
                disp,
                (TARGET_WIDTH, TARGET_HEIGHT),  # 注意：cv2 的顺序是 (width, height)
                interpolation=cv2.INTER_LINEAR
            )
            changed += 1
            if len(changed_examples) < max_examples_print:
                new_h, new_w = disp_resized.shape[:2]
                changed_examples.append((in_path, (h, w), (new_h, new_w)))

        # Step 2: 三通道 -> 单通道，平均
        # 因为每个通道数值一样，平均值等于任一通道；这里严格按“平均值 * 256”
        disp_float = disp_resized.astype(np.float32)
        disp_gray = disp_float.mean(axis=2)              # H x W, float32
        disp_16 = np.round(disp_gray * 256.0)           # 放大 256 倍
        disp_16 = np.clip(disp_16, 0, 65535).astype(np.uint16)  # 转成 uint16

        # Step 3: 写出 16bit 单通道图像
        cv2.imwrite(out_path, disp_16)

        # Step 4: 重新读取新图，验证 bit 数和最大值（仅保存前 max_stats_print 个示例）
        if len(new_stats_examples) < max_stats_print:
            new_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if new_img is None:
                print(f"[WARN] 无法重新读取输出图像: {out_path}")
            else:
                max_new = int(new_img.max())
                new_stats_examples.append((out_path, new_img.dtype, new_img.shape, max_new))

    # 打印统计信息
    print("\n====== 尺寸统计（仅针对 spatial size） ======")
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
            print(f"    {h}x{w} (保持不变，仍然会转换为 16bit 单通道)")

    # 打印原始通道最大值示例
    print("\n====== 原始 24bit 三通道 disparity 示例（通道最大值） ======")
    if not orig_stats_examples:
        print("  无示例（可能没有成功读取到图像）")
    else:
        for p, dtype, shape, ch_max in orig_stats_examples:
            print(f"  {p}")
            print(f"    dtype: {dtype}, shape: {shape}")
            print(f"    通道最大值: C0={ch_max[0]}, C1={ch_max[1]}, C2={ch_max[2]}")

    # 打印新 16bit 单通道示例
    print("\n====== 转换后 16bit 单通道 disparity 示例（最大值 & bit 数） ======")
    if not new_stats_examples:
        print("  无示例（可能写出 / 读入失败）")
    else:
        for p, dtype, shape, maxv in new_stats_examples:
            print(f"  {p}")
            print(f"    dtype: {dtype}, shape: {shape}")
            print(f"    最大值: {maxv}")


if __name__ == "__main__":
    # 原始 24bit 三通道 dense disparity 路径
    input_folder = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST/disp_clean"
    # 新的 16bit 单通道 disparity 输出路径（建议新建目录，避免覆盖原数据）
    output_folder = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST/disp_clean_16bit"

    resize_and_convert_disparity_folder(input_folder, output_folder)
