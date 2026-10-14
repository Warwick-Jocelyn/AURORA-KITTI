import os
import shutil
from pathlib import Path

def replace_images(src_dir: str, target_dir: str) -> int:
    """
    将 src_dir 中的图像文件复制并替换到 target_dir 中，
    如果文件名在 target_dir 的任意子路径中匹配，则执行覆盖复制。
    返回替换的文件数量。
    """
    # 构建 target 中文件名到路径的映射
    target_map = {}
    for root, dirs, files in os.walk(target_dir):
        for fname in files:
            target_map[fname] = os.path.join(root, fname)

    count = 0
    # 迭代 src 中的文件
    for root, dirs, files in os.walk(src_dir):
        for fname in files:
            src_path = os.path.join(root, fname)
            if fname in target_map:
                tgt_path = target_map[fname]
                # 确保目录存在
                os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
                # 复制并覆盖
                shutil.copy2(src_path, tgt_path)
                count += 1
    return count

if __name__ == '__main__':
    # 直接在这里设置源和目标文件夹路径
    src_dir = r"/path/to/aurora-root/Code/rain-rendering/data/output/kitti/data_object/training/rain/200mm/rainy_image_heavy_train"
    target_dir = r"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/frame_image/train"

    replaced = replace_images(src_dir, target_dir)
    print(f"成功替换了 {replaced} 张图像。")
