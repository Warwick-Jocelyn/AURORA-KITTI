import os
import cv2
import pandas as pd
from typing import Iterable, Optional, Union

from lidar_projection_utils import (
    load_from_bin, velo3d_2_camera2d_points,
    create_depth_map, create_intensity_map, create_height_map
)

# ---- 小工具：把 severity 规范化为列表 ----
def _to_list_or_none(val: Optional[Union[str, Iterable[str]]]):
    """
    支持：None / 'light' / ['light','moderate'] / 'light,moderate'
    返回：None 或 列表(list)
    """
    if val is None:
        return None
    if isinstance(val, str):
        # 允许逗号分隔
        items = [s.strip() for s in val.split(",") if s.strip() != ""]
        return items if len(items) > 0 else None
    # 可迭代（list/tuple/set 等）
    try:
        return list(val)
    except TypeError:
        # 其它类型（不支持），退回 None
        return None

def process_lidar_projection_from_csv(
    csv_path,
    raw_lidar_root,
    calib_root,
    depth_out_root,
    intensity_out_root,
    height_out_root,
    split,
    weather=None,
    severity: Optional[Union[str, Iterable[str]]] = None,
    y_filter_min=0,
    y_filter_max=0
):
    df = pd.read_csv(csv_path)

    # 按 weather 过滤（保持你原来的逻辑：仅支持单值；如需多值可同样扩展）
    if weather:
        df = df[df['weather'] == weather]

    # ---- 支持多个 severity 的过滤 ----
    sev_list = _to_list_or_none(severity)
    if sev_list:
        df = df[df['severity'].isin(sev_list)]
    elif isinstance(severity, str) and severity.strip() != "":
        # 单一字符串且未被判定为列表（基本不会走到这里，安全兜底）
        df = df[df['severity'] == severity]

    print(f"🚀 {len(df)} samples to process...")

    generated_count = 0

    for idx, row in df.iterrows():
        name = row['name']
        sequence_id = row['sequence_id']
        date_fields = sequence_id.split('_')
        date_str = '_'.join(date_fields[:3])   # '2011_09_26'

        # ==== 修正点云bin文件名，不包含02或03 ====
        parts = name.split('_image_')
        prefix = parts[0]   # e.g. 2011_09_26_drive_0002_sync
        suffix = parts[1]   # e.g. 02_0000000006
        frame_id = suffix.split('_')[-1]      # 0000000006
        bin_name = f"{prefix}_velodyne_points_{frame_id}.bin"
        bin_path = os.path.join(raw_lidar_root, split, bin_name)
        if not os.path.isfile(bin_path):
            print(f"[Skip] Missing lidar bin: {bin_path}")
            continue

        if '_image_02_' in name:
            cam = '02'
        elif '_image_03_' in name:
            cam = '03'
        else:
            print(f"[Skip] Unknown camera: {name}")
            continue

        # === Calib路径（按每个sequence日期分组） ===
        calib_dir = os.path.join(calib_root, f"{date_str}_calib")
        v2c_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
        c2c_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')

        # === 图像shape，如果已知，直接用定值 ===
        image_shape = (375, 1242, 3)  # 高、宽、通道

        # ==== 投影 ====
        velo = load_from_bin(bin_path)
        ans, _, depth_c, intensity_vals, height_vals = velo3d_2_camera2d_points(
            velo, v_fov=(-24.9, 3.0), h_fov=(-180, 180),
            vc_path=v2c_path, cc_path=c2c_path, mode=cam
        )
        depth_map = create_depth_map(ans, depth_c, image_shape, y_filter_min, y_filter_max)
        intensity_map = create_intensity_map(ans, intensity_vals, image_shape, y_filter_min, y_filter_max)
        height_map = create_height_map(ans, height_vals, image_shape, min_h=-3, max_h=2,
                                       y_filter_min=y_filter_min, y_filter_max=y_filter_max)

        # ==== 输出名和路径（保留02/03相机信息在输出名）====
        base_name = name.replace(f'image_{cam}', f'proj_depth_{cam}')
        depth_out_path = os.path.join(depth_out_root, split, base_name + '.png')
        inten_base_name = name.replace(f'image_{cam}', f'proj_intenisty_{cam}')
        intensity_out_path = os.path.join(intensity_out_root, split, inten_base_name + '.png')
        height_base_name = name.replace(f'image_{cam}', f'proj_height_{cam}')
        height_out_path = os.path.join(height_out_root, split, height_base_name + '.png')

        for path in [depth_out_path, intensity_out_path, height_out_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        cv2.imwrite(depth_out_path, depth_map)
        cv2.imwrite(intensity_out_path, intensity_map)
        cv2.imwrite(height_out_path, height_map)
        print(f"✅ Saved: {os.path.basename(depth_out_path)}, {os.path.basename(intensity_out_path)}, {os.path.basename(height_out_path)}")

        generated_count += 1

    print(f"===> Projected depth maps generated: {generated_count}")
    print(f"===> Total valid rows in CSV: {len(df)}")
    if generated_count != len(df):
        print(f"⚠️ Warning: {len(df)-generated_count} items were skipped (see above for details).")

if __name__ == '__main__':
    SPLIT = 'test'
    base = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset'

    # 这里演示传入多个 severity：
    # 方式1：列表
    sev = ["light", "moderate","heavy"]
    # 方式2：逗号分隔字符串（与方式1等价）
    # sev = "light,moderate"

    process_lidar_projection_from_csv(
        csv_path           = f"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/{SPLIT}.csv",
        raw_lidar_root     = f"{base}/raw_lidar",
        calib_root         = f"{base}/calib",
        depth_out_root     = f"{base}/16bit_proj_depth",
        intensity_out_root = f"{base}/intenisty",
        height_out_root    = f"{base}/height",
        split              = SPLIT,
        weather            = "Snow",
        severity           = sev,    # ← 在这里改成单个或多个都可以
        y_filter_min       = 0,
        y_filter_max       = 130,      # 0 for fog; 150 for snow and rain?
    )

