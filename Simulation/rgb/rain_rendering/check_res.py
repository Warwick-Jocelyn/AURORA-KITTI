import os
from PIL import Image

# 请根据需要修改以下两个路径
folder_a = r"/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2/depth"
folder_b = r"/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/image_2"

# 列出两个文件夹中所有的 PNG 文件
files_a = {f for f in os.listdir(folder_a) if f.lower().endswith('.png')}
files_b = {f for f in os.listdir(folder_b) if f.lower().endswith('.png')}

# 找到同名文件
common_files = files_a & files_b

for filename in sorted(common_files):
    path_a = os.path.join(folder_a, filename)
    path_b = os.path.join(folder_b, filename)

    # 打开图片并获取尺寸
    with Image.open(path_a) as img_a:
        width_a, height_a = img_a.size
    with Image.open(path_b) as img_b:
        width_b, height_b = img_b.size

    # 计算像素总数（分辨率）
    resolution_a = width_a * height_a
    resolution_b = width_b * height_b

    # 如果 A 文件夹中的分辨率大于 B 文件夹，则打印文件名
    if resolution_a > resolution_b:
        print(f"{filename}: A({width_a}x{height_a}) > B({width_b}x{height_b})")
