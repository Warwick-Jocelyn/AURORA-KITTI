# import argparse
# from pathlib import Path
# from PIL import Image, ImageChops
# import numpy as np
# import sys

# # --- 核心设置 ---
# TARGET_W, TARGET_H = 1216, 352

# # --- 1. 复制你提供的核心裁剪计算逻辑 ---
# def compute_crop_box(w, h):
#     """
#     计算裁剪框 (left, top, right, bottom)
#     策略: 
#     - 高度: 保留底部 (Top cut) -> top = h - TARGET_H
#     - 宽度: 左右居中 (Centered)
#     """
#     if h < TARGET_H or w < TARGET_W:
#         return None
    
#     # 高度处理：保留底部，切掉顶部
#     top = h - TARGET_H
#     bottom = h
    
#     # 宽度处理：居中
#     to_crop_w = w - TARGET_W
#     left = to_crop_w // 2
#     right_extra = to_crop_w - left  # 处理奇数情况
#     right = w - right_extra
    
#     # 确保坐标不越界 (Clamp)
#     left = max(0, min(left, w - TARGET_W))
#     right = max(left + TARGET_W, min(w, right))
#     top = max(0, min(top, h - TARGET_H))
#     bottom = max(top + TARGET_H, min(h, bottom))
    
#     return (left, top, right, bottom)

# # --- 2. 单张图片加载与处理函数 ---
# def load_and_process_image(image_path):
#     """
#     读取图片，如果尺寸不对则进行 crop，最后统一转为 RGB 用于对比
#     """
#     path_obj = Path(image_path)
#     if not path_obj.exists():
#         raise FileNotFoundError(f"无法找到文件: {image_path}")

#     # 打开图片
#     img = Image.open(path_obj)
#     w, h = img.size
    
#     print(f"  -> 读取图像: {path_obj.name} | 原始尺寸: {w}x{h}")

#     # 判断是否需要裁剪
#     if (w, h) == (TARGET_W, TARGET_H):
#         print(f"     尺寸符合要求，无需裁剪。")
#         processed_img = img
#     else:
#         box = compute_crop_box(w, h)
#         if box is None:
#             raise ValueError(f"     ❌ 错误: 图片尺寸 ({w}x{h}) 小于目标尺寸 ({TARGET_W}x{TARGET_H})，无法裁剪。")
        
#         print(f"     执行裁剪 Box: {box} (保留底部，水平居中)")
#         processed_img = img.crop(box)

#     # 统一转换为 RGB 模式以确保像素对比公平 (忽略 Alpha 通道差异)
#     if processed_img.mode != "RGB":
#         processed_img = processed_img.convert("RGB")
        
#     return processed_img

# # --- 3. 对比主逻辑 ---
# def compare_two_images_with_crop(path_a, path_b):
#     print("=" * 60)
#     print(f"开始对比任务:\n A: {path_a}\n B: {path_b}")
#     print("-" * 60)

#     try:
#         # 步骤 1: 处理两张图
#         img_a = load_and_process_image(path_a)
#         img_b = load_and_process_image(path_b)
        
#         # 步骤 2: 转换为 Numpy 数组进行计算
#         arr_a = np.array(img_a)
#         arr_b = np.array(img_b)

#         print("-" * 60)
        
#         # 步骤 3: 形状对比
#         if arr_a.shape != arr_b.shape:
#             print("❌ 结果: 形状不一致！")
#             print(f"   图A形状: {arr_a.shape}")
#             print(f"   图B形状: {arr_b.shape}")
#             return

#         # 步骤 4: 像素值严格对比
#         # 计算差值图
#         diff = np.abs(arr_a.astype(int) - arr_b.astype(int)).astype(np.uint8)
        
#         # 如果 diff 中所有值都是 0，则完全一致
#         is_identical = np.all(diff == 0)
        
#         if is_identical:
#             print("✅ 结果: 两张图片经过处理后【完全一致】！")
#         else:
#             # 计算有多少个像素不同
#             # np.count_nonzero(diff) 会计算 R,G,B 三个通道所有的非0值
#             # 如果我们想算“有多少个像素点不一样”（不管这个点是R不同还是G不同），需要先合并通道
#             diff_gray = np.sum(diff, axis=2) # 求和，只要RGB有一个不为0，和就不为0
#             diff_pixel_count = np.count_nonzero(diff_gray)
#             total_pixels = TARGET_W * TARGET_H
            
#             print("❌ 结果: 两张图片内容【不一致】。")
#             print(f"   不一样像素数量: {diff_pixel_count} / {total_pixels} ({diff_pixel_count/total_pixels:.2%})")
            
#             # 保存差异图
#             diff_image = Image.fromarray(diff)
#             save_name = "diff_crop_result.png"
#             diff_image.save(save_name)
#             print(f"   差异图像已保存为: {save_name}")

#     except Exception as e:
#         print(f"\n❌ 程序执行出错: {e}")

# # --- 入口 ---
# if __name__ == "__main__":
#     #在此处填入你的图片路径
#     IMAGE_1 = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/image_clean_original"
#     IMAGE_2 = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/image_clean"
    
#     # 为了演示，如果路径不存在会提示修改
#     if Path(IMAGE_1).name == "image_1.png": 
#         print("⚠️ 请在代码末尾修改 IMAGE_1 和 IMAGE_2 为真实的图片路径。")
#     else:
#         compare_two_images_with_crop(IMAGE_1, IMAGE_2)

import argparse
from pathlib import Path
from PIL import Image

# --- 核心设置 ---
TARGET_W, TARGET_H = 1216, 352


def compute_crop_box(w, h):
    """
    计算裁剪框 (left, top, right, bottom)
    策略:
    - 高度: 保留底部 -> top = h - TARGET_H
    - 宽度: 左右居中
    """
    if h < TARGET_H or w < TARGET_W:
        return None

    # 高度处理：保留底部，切掉顶部
    top = h - TARGET_H
    bottom = h

    # 宽度处理：居中
    to_crop_w = w - TARGET_W
    left = to_crop_w // 2
    right_extra = to_crop_w - left  # 处理奇数情况
    right = w - right_extra

    # Clamp 确保不越界
    left = max(0, min(left, w - TARGET_W))
    right = max(left + TARGET_W, min(w, right))
    top = max(0, min(top, h - TARGET_H))
    bottom = max(top + TARGET_H, min(h, bottom))

    return (left, top, right, bottom)


def process_one_image(in_path: Path, out_path: Path, on_small: str = "skip"):
    """
    读取图片 -> 필요时 crop -> 转 RGB -> 保存到 out_path
    """
    img = Image.open(in_path)
    w, h = img.size

    # 不需要裁剪
    if (w, h) == (TARGET_W, TARGET_H):
        processed = img
    else:
        box = compute_crop_box(w, h)
        if box is None:
            msg = f"图片尺寸({w}x{h})小于目标({TARGET_W}x{TARGET_H})，无法裁剪: {in_path}"
            if on_small == "error":
                raise ValueError(msg)
            else:
                print(f"  [SKIP] {msg}")
                return False
        processed = img.crop(box)

    # 统一转为 RGB
    if processed.mode != "RGB":
        processed = processed.convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 尽量保持原始格式（由后缀决定）
    # 注意：RGB保存成jpg/png都没问题；如果你想强制png可自行改后缀
    processed.save(out_path)

    return True


def iter_images(input_dir: Path, recursive: bool, exts: set[str]):
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in input_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def main():
    parser = argparse.ArgumentParser(
        description="Batch crop RGB images to 1216x352 (keep bottom, horizontally centered)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹路径")
    parser.add_argument(
        "--recursive", action="store_true", help="递归处理子文件夹，并在输出中保留目录结构"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
        help="要处理的图片后缀，逗号分隔（默认: .png,.jpg,.jpeg,.bmp,.tif,.tiff）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在则覆盖（默认不覆盖，直接跳过）",
    )
    parser.add_argument(
        "--on_small",
        choices=["skip", "error"],
        default="skip",
        help="遇到尺寸小于目标的图片怎么处理：skip=跳过，error=直接报错退出（默认skip）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
            for e in args.ext.split(",") if e.strip()}

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir 不存在或不是文件夹: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_images(input_dir, args.recursive, exts))
    if not files:
        print(f"未找到匹配后缀的图片: exts={sorted(exts)} in {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {TARGET_W}x{TARGET_H}")
    print(f"递归: {args.recursive} | 覆盖: {args.overwrite} | on_small: {args.on_small}")
    print(f"待处理图片数量: {len(files)}")
    print("-" * 60)

    ok = 0
    skipped_exist = 0
    skipped_small = 0
    failed = 0

    for i, in_path in enumerate(files, 1):
        # 保持相对路径（recursive时保留子目录结构）
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel  # 文件名保持不变

        if out_path.exists() and not args.overwrite:
            skipped_exist += 1
            continue

        try:
            done = process_one_image(in_path, out_path, on_small=args.on_small)
            if done:
                ok += 1
                if i % 200 == 0:
                    print(f"  processed {i}/{len(files)} ...")
            else:
                skipped_small += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {in_path} -> {e}")

    print("-" * 60)
    print("完成！统计：")
    print(f"  成功处理: {ok}")
    print(f"  跳过(已存在且不覆盖): {skipped_exist}")
    print(f"  跳过(尺寸太小): {skipped_small}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()

# python crop_RGB.py \
#   --input_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image_clean_original \
#   --output_dir /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image_clean \
#   --recursive
