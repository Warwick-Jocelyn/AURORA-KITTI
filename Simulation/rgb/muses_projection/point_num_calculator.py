import os
import numpy as np
import cv2

def count_points_in_png(directory):
    """
    遍历指定目录及其所有子目录，统计所有 16-bit 单通道 PNG 图像中的有效点数量。
    
    :param directory: 目标目录
    """
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    total_valid_points = 0
    num_images = len(png_files)
    
    for file_path in png_files:
        try:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is not None and len(image.shape) == 2:
                valid_points = np.count_nonzero(image)
                total_valid_points += valid_points
        except Exception as e:
            print(f'Error reading {file_path}: {str(e)}')
    
    avg_valid_points = total_valid_points / num_images if num_images > 0 else 0
    
    print(f"Total number of png files: {num_images}")
    print(f"Total number of valid points: {total_valid_points}")
    print(f"Average number of valid points per image: {avg_valid_points:.2f}")
    
    return png_files

# 示例使用
if __name__ == "__main__":
    directory = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps_16bit/val/snow/"  # 替换为你的文件夹路径
    count_points_in_png(directory)

# ---------------Clear-----------
# Train：Total number of npy files: 450
# Total number of points: 19070220 = 42378.26

# Test 
# Total number of png files: 225
# Total number of valid points: 9432128
# Average number of valid points per image: 41920.57

#val
# Total number of png files: 75
# Total number of valid points: 3226129
# Average number of valid points per image: 43015.05

# ---------------fog---------------
# Train
# Total number of png files: 350
# Total number of valid points: 9622350
# Average number of valid points per image: 27492.43

# Test 
# Total number of png files: 175
# Total number of valid points: 4778887
# Average number of valid points per image: 27307.93

#val
# Total number of png files: 58
# Total number of valid points: 1529482
# Average number of valid points per image: 26370.38

# ---------------rain---------------
# Train
# Total number of png files: 350
# Total number of valid points: 12498528
# Average number of valid points per image: 35710.08

# Test 
# Total number of png files: 175
# Total number of valid points: 6034636
# Average number of valid points per image: 34483.63

#val
# Total number of png files: 59
# Total number of valid points: 1926588
# Average number of valid points per image: 32654.03
# ----------------snow---------------
# Train
# Total number of png files: 350
# Total number of valid points: 15098888
# Average number of valid points per image: 43139.68

# Test 
# Total number of png files: 175
# Total number of valid points: 7237972
# Average number of valid points per image: 41359.84

#val
# Total number of png files: 58
# Total number of valid points: 2528173
# Average number of valid points per image: 43589.19