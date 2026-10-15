import os
import shutil
import pandas as pd

# ———— 配置区域 ————
# CSV 文件路径
csv_file = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/val.csv'

# 源文件夹
rgb_src_dir   = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/frame_image/val'
depth_src_dir = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/8bit_clean_depth/val'
calib_src_dir = '/path/to/aurora-root/Code/rain-rendering/calib_for_rain_heavy_light/calib'

# 目标文件夹（自动创建）
rgb_dst_dir   = '/path/to/aurora-root/Code/rain-rendering/data_for_rain_heavy/val/image_2'
depth_dst_dir = '/path/to/aurora-root/Code/rain-rendering/data_for_rain_heavy/val/image_2/depth'
calib_dst_dir = '/path/to/aurora-root/Code/rain-rendering/data_for_rain_heavy/val/calib'

# ———— 脚本主体 ————

# 1. 创建目标文件夹
for d in [rgb_dst_dir, depth_dst_dir, calib_dst_dir]:
    os.makedirs(d, exist_ok=True)

# 2. 读取 CSV 并过滤出 weather 为 "Rain" 且 severity 为 "heavy" 的行
df = pd.read_csv(csv_file)
filtered = df[
    (df['weather'].str.lower() == 'rain') &
    (df['severity'].str.lower() == 'heavy')
]

# 3. 对每一行执行复制操作
for _, row in filtered.iterrows():
    seq_id = row['sequence_id']    # e.g. '2011_09_26_drive_0002_sync'
    name   = row['name']           # e.g. '..._image_02_0000000015'

    # ---- 复制 RGB 图像 ----
    src_rgb = os.path.join(rgb_src_dir, f"{name}.png")
    dst_rgb = os.path.join(rgb_dst_dir, f"{name}.png")
    shutil.copy2(src_rgb, dst_rgb)

    # ---- 复制并重命名 深度图像 ----
    # 将 name 中的 '_image_' 替换为 '_8bitdepth_' 得到源深度图名
    depth_name = name.replace('_image_', '_8bitdepth_')
    src_depth = os.path.join(depth_src_dir, f"{depth_name}.png")
    # 目标文件名恢复成原始的 '<name>.png'
    dst_depth = os.path.join(depth_dst_dir, f"{name}.png")
    shutil.copy2(src_depth, dst_depth)

    # ---- 复制并重命名 标定 txt ----
    # 提取日期前缀（YYYY_MM_DD）
    date_prefix = seq_id.split('_drive')[0]
    calib_filename = f"{date_prefix}_calib.txt"
    src_calib = os.path.join(calib_src_dir, calib_filename)
    # 目标 txt 名称与图片同名，只是扩展名为 .txt
    dst_calib = os.path.join(calib_dst_dir, f"{name}.txt")
    shutil.copy2(src_calib, dst_calib)

    print(f"Processed: {name}")

print("所有符合条件的文件已复制完毕。")
