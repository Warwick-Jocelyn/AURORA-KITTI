import os
import cv2
import numpy as np

def process_images(root_folder):
    global_min_row = float('inf')  # 初始化全局最小值为正无穷
    global_max_row = float('-inf')  # 初始化全局最大值为负无穷

    for folder_path, sub_folders, file_names in os.walk(root_folder):
        for file_name in file_names:
            if file_name.endswith(('.png', '.jpg', '.tif')):  # 支持的扩展名
                file_path = os.path.join(folder_path, file_name)
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # 加载为16位单通道图像

                if image is None:
                    print(f"无法读取文件 {file_name}，跳过。")
                    continue

                # 获取所有非零点（有效点）的坐标
                valid_points = np.argwhere(image > 0)

                if valid_points.size == 0:
                    print(f"文件 {file_name} 中没有有效点，跳过。")
                    continue

                # 获取当前图像的 row 的最小值和最大值
                min_row = valid_points[:, 0].min()
                max_row = valid_points[:, 0].max()

                # 更新全局最小值和最大值
                global_min_row = min(global_min_row, min_row)
                global_max_row = max(global_max_row, max_row)

                print(f"文件: {file_name}")
                # print(f"  当前 row 的最小值: {min_row}")
                # print(f"  当前 row 的最大值: {max_row}")
                print("-" * 50)

    # 输出全局最小值和最大值
    if global_min_row == float('inf') or global_max_row == float('-inf'):
        print("未找到任何有效点。")
    else:
        print("全局统计结果:")
        print(f"  row 的全局最小值: {global_min_row}")
        print(f"  row 的全局最大值: {global_max_row}")

# 替换为包含图像的主文件夹路径
root_folder = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps_16bit"
process_images(root_folder)

# 全局统计结果:
#   row 的全局最小值: 146
#   row 的全局最大值: 855