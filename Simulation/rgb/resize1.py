import os
from pathlib import Path
import cv2
from tqdm import tqdm

def resize_png_folder_cv2(src_root: str, dst_root: str,
                          src_size=(1280, 384), dst_size=(1242, 375)):
    """
    遍历 src_root 下所有 .png 文件，将它们从 src_size 缩放到 dst_size，
    并保存到 dst_root 下相同的相对路径。使用 OpenCV 的 INTER_AREA 插值，
    并显示 tqdm 进度条。
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 收集所有 PNG 文件路径
    all_pngs = list(src_root.rglob('*.png'))

    # 在进度条中逐个处理
    for src_path in tqdm(all_pngs, desc="Resizing images", unit="file"):
        # 计算相对路径并准备输出目录
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取图像（BGR 格式）
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # 无法读取则跳过
            continue

        # 使用 INTER_AREA 插值进行高质量缩小
        resized = cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_path), resized)

    print(f"🎉 Done! All {len(all_pngs)} images resized and saved to:\n  {dst_root}")

if __name__ == "__main__":
    # 直接在这里设置输入输出路径
    SRC_PATH = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/RGBL_Aligned/raingan"
    DST_PATH = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/RGBL_Aligned/raingan"

    resize_png_folder_cv2(SRC_PATH, DST_PATH)
