from pathlib import Path
import cv2
from tqdm import tqdm

def resize_png_folder_cv2(src_root: str, dst_root: str, dst_size=(1216, 352)):
    """
    遍历 src_root 下所有 .png 文件，将它们从任意尺寸缩放到 dst_size，
    并保存到 dst_root 下相同的相对路径。使用 OpenCV 的 INTER_AREA 插值，
    并显示 tqdm 进度条。
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    all_pngs = list(src_root.rglob("*.png"))

    for src_path in tqdm(all_pngs, desc="Resizing images", unit="file"):
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # 任意输入尺寸 -> 固定输出尺寸
        resized = cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)

        # 可选：确保输出目录存在后写入
        ok = cv2.imwrite(str(dst_path), resized)
        if not ok:
            # 写入失败就跳过（也可以打印日志）
            continue

    print(f"🎉 Done! All {len(all_pngs)} images resized and saved to:\n  {dst_root}")

if __name__ == "__main__":
    SRC_PATH = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/image_clean_original"
    DST_PATH = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL/image_clean_original"

    resize_png_folder_cv2(SRC_PATH, DST_PATH, dst_size=(1242, 375))
