import os
import numpy as np
import imageio.v2 as imageio

# 输入与输出路径
input_dir = "/path/to/aurora-root/PhD_Project_2025/BP-Net/datas/kitti/val_selection_cropped_TEST/velodyne_raw"
output_dir = "/path/to/aurora-root/PhD_Project_2025/BP-Net/datas/kitti/val_selection_cropped_TEST/velodyne_raw_npy_in_meter"

os.makedirs(output_dir, exist_ok=True)

# 遍历所有png文件
for fname in os.listdir(input_dir):
    if fname.endswith(".png"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".png", ".npy"))
        
        # 读取16bit深度图
        depth_uint16 = imageio.imread(in_path).astype(np.float32)
        
        # 转换为米（float）
        depth_m = depth_uint16 / 256.0
        
        # 保存为npy
        np.save(out_path, depth_m)
        
        print(f"Converted: {fname} -> {out_path}")

print("✅ All PNG files converted to .npy (in meters).")
