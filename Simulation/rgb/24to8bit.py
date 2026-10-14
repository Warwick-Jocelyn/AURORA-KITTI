import os
from PIL import Image

def convert_rgb_to_gray(root_dir):
    """
    遍历 root_dir 下所有子目录，找到所有 .png 文件，
    将 RGB（伪三通道）图像转换为单通道 8-bit 灰度图，并覆盖保存。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.png'):
                continue

            full_path = os.path.join(dirpath, fname)
            try:
                with Image.open(full_path) as img:
                    # 只处理伪三通道图像（每个通道值相同的 RGB）
                    if img.mode == 'RGB':
                        # 拆分通道，取 R 通道即可（R=G=B）
                        r, g, b = img.split()

                        # 可选：检查三个通道是否真的相同
                        # if list(r.getdata()) != list(g.getdata()):
                        #     print(f"Warning: channels differ in {full_path}")
                        
                        # 保存为单通道灰度图（模式 'L'）
                        gray = r.convert('L')
                        gray.save(full_path)
                        print(f"Converted: {full_path}")

            except Exception as e:
                print(f"Failed to process {full_path}: {e}")

if __name__ == '__main__':
    # 把下面路径改成你的顶层目录
    root_directory = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/8bit_clean_depth'
    convert_rgb_to_gray(root_directory)
