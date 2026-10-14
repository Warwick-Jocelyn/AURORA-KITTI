import glob
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt 

# 假设已有load_from_bin函数，用于加载 .bin 格式的点云数据
def load_from_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points  # shape: (N, 4) - [x, y, z, reflectance]

def depth_color(val, min_d=0, max_d=120):
    """
    根据距离（米）映射HSV中H的值，
    近处为红色，远处为蓝色
    （此函数仅用于显示目的）
    """    
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

def in_h_range_points(points, m, n, fov):
    return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180),
                          np.arctan2(n, m) < (-fov[0] * np.pi / 180))

def in_v_range_points(points, m, n, fov):
    return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180),
                          np.arctan2(n, m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """根据水平和垂直FOV筛选点"""
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]

def in_range_points(points, size):
    return np.logical_and(points >= 0, points < size)

# ------------ 过滤函数 ------------
def velo_points_filter(points_xyz, v_fov, h_fov, return_mask=False):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)

    h_mask = in_h_range_points(points_xyz, x, y, h_fov)
    v_mask = in_v_range_points(points_xyz, dist, z, v_fov)
    mask   = np.logical_and(h_mask, v_mask)

    xyz = points_xyz[mask]
    dist_sel = dist[mask]

    xyz_hom = np.vstack((xyz.T, np.ones((1, xyz.shape[0]))))
    color = depth_color(dist_sel, 0, 70)

    if return_mask:
        return xyz_hom, color, mask
    return xyz_hom, color

def calib_velo2cam0(filepath):
    """读取 Velodyne → cam0 的外参
    读取 velodyne 到相机的外参，返回旋转矩阵 R (3x3) 和平移向量 T (3x1)"""
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, val = line.strip().split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ').reshape(3, 3)
            elif key == 'T':
                T = np.fromstring(val, sep=' ').reshape(3, 1)
    RT = np.hstack((R, T))
    return RT

def calib_cam2cam(filepath, mode):
    """
    获取 rectified 投影矩阵 P 和图像矫正旋转矩阵 R_rect_00
    """
    P_rect = None
    R_rect = None
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, val = line.strip().split(':', 1)
            if key == f'P_rect_{mode}':
                P_rect = np.fromstring(val, sep=' ').reshape(3, 4)
            elif key == f'R_rect_00':
                R_rect = np.fromstring(val, sep=' ').reshape(3, 3)
    return R_rect, P_rect

# ---------------------- 核心函数 ---------------------------
# ------------ 主入口 Updated ------------
def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02'):
    RT = calib_velo2cam0(vc_path)
    R_rect, P_rect = calib_cam2cam(cc_path, mode)
    xyz_velo, color, mask = velo_points_filter(points[:, :3], v_fov, h_fov, return_mask=True)
    xyz_cam0 = RT @ xyz_velo
    xyz_rect = R_rect @ xyz_cam0
    depth_c  = xyz_rect[2, :].copy()
    uv_hom = P_rect @ np.vstack((xyz_rect, np.ones((1, xyz_rect.shape[1]))))
    uv     = uv_hom[:2, :] / uv_hom[2, :]
    intensity = np.clip(points[mask, 3], 0.0, 1.0)
    height    = points[mask, 2]
    return uv, color, depth_c, intensity, height

def create_depth_map(points, depth_values, image_shape, y_filter_min=0, y_filter_max=0):
    height, width = image_shape[:2]
    depth_map = np.zeros((height, width), dtype=np.uint16)
    for i in range(points.shape[1]):
        x = int(points[0, i])
        y = int(points[1, i])
        if 0 <= x < width and 0 <= y < height and (y < y_filter_min or y >= y_filter_max):
            # KITTI 16-bit depth convention: uint16 = meters * 256 (load with /256).
            # NOTE: the existing on-disk weather + clean sparse sets were generated at
            # *255 and are kept that way for mutual consistency; this *256 fix applies
            # to FUTURE regenerations only.
            new_depth = depth_values[i] * 256.0
            if depth_map[y, x] == 0 or new_depth < depth_map[y, x]:
                depth_map[y, x] = new_depth
    return depth_map

def create_intensity_map(points, intensity_values, image_shape, y_filter_min=0, y_filter_max=0):
    height, width = image_shape[:2]
    intensity_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(points.shape[1]):
        x = int(points[0, i])
        y = int(points[1, i])
        if 0 <= x < width and 0 <= y < height and (y < y_filter_min or y >= y_filter_max):
            val = int(intensity_values[i] * 255)
            if intensity_map[y, x] == 0 or val > intensity_map[y, x]:
                intensity_map[y, x] = val
    return intensity_map

def create_height_map(points, height_values, image_shape, min_h=-3, max_h=1, y_filter_min=0, y_filter_max=0):
    height, width = image_shape[:2]
    height_map = np.zeros((height, width), dtype=np.uint8)
    norm_height = np.clip((height_values - min_h) / (max_h - min_h), 0, 1)
    norm_height = (norm_height * 255).astype(np.uint8)
    for i in range(points.shape[1]):
        x = int(points[0, i])
        y = int(points[1, i])
        if 0 <= x < width and 0 <= y < height and (y < y_filter_min or y >= y_filter_max):
            val = norm_height[i]
            if height_map[y, x] == 0 or val > height_map[y, x]:
                height_map[y, x] = val
    return height_map
