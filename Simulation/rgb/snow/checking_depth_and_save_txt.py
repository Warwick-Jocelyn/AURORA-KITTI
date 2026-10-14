# import numpy as np
# img_name = 'SIM_THI_Car_28m'
# depth = np.loadtxt('input/' + img_name +'.txt')

# print(depth)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

# =========== 修改这里即可 =============
# input_img_path = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/0000000006_clear.png'     # 彩色原图（BGR）
# input_depth_path = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/depth_estimator.png'  # 16bit单通道深度图，单位毫米

input_img_path = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/0000000006_clear.png'     # 彩色原图（BGR）
input_depth_path = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/depth_estimator.png'  # 16bit单通道深度图，单位毫米

# 读取16bit深度图（norm_depth），需要还原成实际深度（米）
norm_depth = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)
if norm_depth is None:
    raise FileNotFoundError(f"Depth map not found: {input_depth_path}")
depth = norm_depth.astype(np.float32) / 255.0 * 200   # 还原为float米
# depth = norm_depth.astype(np.float32)

# print(depth[200][700])
print(depth)

# 在这里指定你要保存的 txt 路径
output_txt_path = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/input_depth_anything/depth.txt'
os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

# # 方法1：使用 numpy.savetxt
# #   fmt 可以根据需要调整小数位数，delimiter=' ' 表示用空格分隔
# np.savetxt(output_txt_path, depth, fmt='%.6f', delimiter=' ')
# print(f"Depth 已保存到：{output_txt_path}")

# 方法2：手动写入，逐行拼接
with open(output_txt_path, 'w') as f:
    for row in depth:
        # map(str, row) 将每个数转换为字符串，用空格连接
        f.write(' '.join(map(str, row)) + '\n')
print(f"Depth 已保存到：{output_txt_path}")
