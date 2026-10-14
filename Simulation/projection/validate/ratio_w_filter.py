import os
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union
from collections import defaultdict

# ========== 基础工具 ==========
def _to_list_or_none(val: Optional[Union[str, Iterable[str]]]):
    """
    None / 'light' / ['light','moderate'] / 'light,moderate' -> list 或 None
    """
    if val is None:
        return None
    if isinstance(val, str):
        items = [s.strip() for s in val.split(",") if s.strip() != ""]
        return items if items else None
    try:
        return list(val)
    except TypeError:
        return None

def read_kitti_bin_points(bin_file_path: str) -> np.ndarray:
    """
    读取 KITTI 格式的 .bin 点云，返回 (N, 4) 的数组 [x, y, z, reflectance]
    """
    arr = np.fromfile(bin_file_path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid .bin size for {bin_file_path}")
    return arr.reshape(-1, 4)

def build_bin_path_from_name(name: str, raw_lidar_root: str, split: str) -> str:
    """
    按你之前的规则，从 image 名字生成 velodyne .bin 路径
    例：
      name: 2011_09_26_drive_0002_sync_image_02_0000000006
      ->  2011_09_26_drive_0002_sync_velodyne_points_0000000006.bin
    """
    parts = name.split('_image_')
    if len(parts) != 2:
        return ""  # 兜底
    prefix = parts[0]                      # e.g. 2011_09_26_drive_0002_sync
    suffix = parts[1]                      # e.g. 02_0000000006
    frame_id = suffix.split('_')[-1]       # 0000000006
    bin_name = f"{prefix}_velodyne_points_{frame_id}.bin"
    return os.path.join(raw_lidar_root, split, bin_name)

# ========== 主功能 ==========
def compute_range_bin_proportions(
    csv_path: str,
    raw_lidar_root: str,
    split: str,
    weather: Optional[str] = None,
    severity: Optional[Union[str, Iterable[str]]] = None,
    range_bins=((0.0, 20.0), (20.0, 40.0), (40.0, None)),
    use_euclidean_distance: bool = True,
    # 新增：高度上限（米）。为 None 则不限制
    height_cap: Optional[float] = None,
):
    """
    读取 CSV，按 weather / severity 过滤后，
    （可选）先按 z ≤ height_cap 过滤，再统计距离段比例。
    同时计算：
      - Average of points（每帧平均点数）
      - Average point distance（所有点的平均欧氏距离）
    """
    df = pd.read_csv(csv_path)

    if weather:
        df = df[df['weather'] == weather]

    sev_list = _to_list_or_none(severity)
    if sev_list:
        df = df[df['severity'].isin(sev_list)]

    print(f"🗂️ Split={split} | weather={weather} | severity={sev_list} | candidates={len(df)}")
    if height_cap is not None:
        print(f"   Height constrain: z ≤ {height_cap:.6f} m")

    total_counts = np.zeros(len(range_bins), dtype=np.int64)
    total_points = 0                   # 过滤后，所有帧的总点数
    skipped = 0
    valid_samples = 0                  # 成功读取 .bin 的帧数（即使过滤后为 0 点也计入）
    sum_dist_over_all_points = 0.0     # 过滤后所有点的距离总和（用于 avg point distance）

    for _, row in df.iterrows():
        name = row['name']
        bin_path = build_bin_path_from_name(name, raw_lidar_root, split)
        if not bin_path or not os.path.isfile(bin_path):
            skipped += 1
            continue

        try:
            pts = read_kitti_bin_points(bin_path)  # (N,4)
        except Exception as e:
            print(f"[Skip] {bin_path}: {e}")
            skipped += 1
            continue

        # —— 高度限制 ——
        if height_cap is not None:
            pts = pts[pts[:, 2] <= height_cap]

        valid_samples += 1
        if pts.shape[0] == 0:
            # 空帧：参与“平均每帧点数”的分母，但不会贡献点与距离
            continue

        # 距离
        if use_euclidean_distance:
            dists = np.linalg.norm(pts[:, :3], axis=1)
        else:
            dists = np.abs(pts[:, 0])

        # 累加用于“平均点距”和“平均每帧点数”
        sum_dist_over_all_points += float(dists.sum())
        total_points += int(dists.size)

        # 桶计数
        bin_counts = np.zeros(len(range_bins), dtype=np.int64)
        for i, (lo, hi) in enumerate(range_bins):
            if hi is None:
                mask = dists >= lo
            else:
                mask = (dists >= lo) & (dists < hi)
            bin_counts[i] = int(mask.sum())

        total_counts += bin_counts

    if total_points == 0 or valid_samples == 0:
        print("❗没有有效点云可统计（可能 .bin 缺失、或高度约束过滤过严）")
        return {
            'total_points': 0,
            'bin_counts': [0]*len(range_bins),
            'bin_props': [0.0]*len(range_bins),
            'skipped': skipped,
            'num_samples': valid_samples,
            'avg_points_per_frame': 0.0,
            'avg_point_distance': 0.0,
        }

    bin_props = (total_counts / total_points).tolist()
    avg_points_per_frame = total_points / valid_samples
    avg_point_distance = (sum_dist_over_all_points / total_points) if total_points > 0 else 0.0

    result = {
        'total_points': int(total_points),
        'bin_counts': total_counts.tolist(),
        'bin_props': bin_props,
        'skipped': skipped,
        'num_samples': valid_samples,
        'avg_points_per_frame': float(avg_points_per_frame),
        'avg_point_distance': float(avg_point_distance),
    }

    # 打印简报
    labels = [f"[{lo},{'∞' if hi is None else hi}) m" for (lo, hi) in range_bins]
    print(f"✅ 有效样本: {valid_samples} | 跳过: {skipped}")
    for lab, c, p in zip(labels, result['bin_counts'], result['bin_props']):
        print(f"  {lab:>10}: {c:>10d} pts | {p*100:6.2f}%")
    print(f"  {'TOTAL':>10}: {result['total_points']:>10d} pts")
    print(f"  Average of points     : {avg_points_per_frame:.6f} pts/frame")
    print(f"  Average point distance: {avg_point_distance:.6f} m")

    return result

def compute_range_bin_table_by_buckets(
    csv_path: str,
    raw_lidar_root: str,
    split: str,
    bucket_fields=('weather', 'severity'),
    range_bins=((0.0, 20.0), (20.0, 40.0), (40.0, None)),
    use_euclidean_distance: bool = True,
    # 高度上限（米）。为 None 则不限制
    height_cap: Optional[float] = None,
) -> pd.DataFrame:
    """
    分桶统计；若提供 height_cap，则每帧先做 z ≤ height_cap 过滤。
    额外输出：
      - avg_points_per_frame（该桶内每帧平均点数）
      - avg_point_distance（该桶内所有点的平均欧氏距离）
    """
    df = pd.read_csv(csv_path)
    rows = []
    grouped = df.groupby(list(bucket_fields), dropna=False)

    if height_cap is not None:
        print(f"   Height constrain: z ≤ {height_cap:.6f} m")

    for bucket_vals, subdf in grouped:
        if not isinstance(bucket_vals, tuple):
            bucket_vals = (bucket_vals,)

        total_counts = np.zeros(len(range_bins), dtype=np.int64)
        total_points = 0
        skipped = 0
        valid_samples = 0
        sum_dist_over_all_points = 0.0

        for _, row in subdf.iterrows():
            bin_path = build_bin_path_from_name(row['name'], raw_lidar_root, split)
            if not bin_path or not os.path.isfile(bin_path):
                skipped += 1
                continue

            try:
                pts = read_kitti_bin_points(bin_path)
            except Exception:
                skipped += 1
                continue

            if height_cap is not None:
                pts = pts[pts[:, 2] <= height_cap]

            valid_samples += 1
            if pts.shape[0] == 0:
                continue

            if use_euclidean_distance:
                dists = np.linalg.norm(pts[:, :3], axis=1)
            else:
                dists = np.abs(pts[:, 0])

            sum_dist_over_all_points += float(dists.sum())
            total_points += int(dists.size)

            bin_counts = np.zeros(len(range_bins), dtype=np.int64)
            for i, (lo, hi) in enumerate(range_bins):
                if hi is None:
                    mask = dists >= lo
                else:
                    mask = (dists >= lo) & (dists < hi)
                bin_counts[i] = int(mask.sum())
            total_counts += bin_counts

        props = (total_counts / total_points).tolist() if total_points > 0 else [0.0]*len(range_bins)
        avg_points_per_frame = (total_points / valid_samples) if valid_samples > 0 else 0.0
        avg_point_distance = (sum_dist_over_all_points / total_points) if total_points > 0 else 0.0

        row_dict = {k: v for k, v in zip(bucket_fields, bucket_vals)}
        for i, (lo, hi) in enumerate(range_bins):
            lab = f"[{lo},{'inf' if hi is None else hi})m"
            row_dict[f"count{lab}"] = int(total_counts[i])
            row_dict[f"prop{lab}"]  = float(props[i])

        row_dict["total_points"] = int(total_points)
        row_dict["num_samples"]  = int(valid_samples)
        row_dict["skipped"]      = int(skipped)
        row_dict["avg_points_per_frame"] = float(avg_points_per_frame)
        row_dict["avg_point_distance"]   = float(avg_point_distance)
        rows.append(row_dict)

    out_df = pd.DataFrame(rows).sort_values(list(bucket_fields)).reset_index(drop=True)
    return out_df

# ======== 使用示例 ========
if __name__ == "__main__":
    # height_cap_value = 2.913000  # ← 来自 max_height_in_folder.py 的输出
    height_cap_value = 3.5
    SPLIT = "val"
    base  = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    csv_path       = f"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/{SPLIT}.csv"
    raw_lidar_root = f"{base}/raw_lidar"
    # raw_lidar_root = f"/path/to/aurora-root/PhD_Project_2025/lidar_compound_noise_models_original/raw_lidar_simulated_Fog/{SPLIT}"

    # 1) 针对单一筛选（例如：Clear + severity in ['None']）
    res = compute_range_bin_proportions(
        csv_path=csv_path,
        raw_lidar_root=raw_lidar_root,
        split=SPLIT,
        # weather="Clear",
        # severity=[" None"], # tab None
        weather="Snow",
        severity=[ "light", "heavy"], # tab None
        range_bins=((0.0, 12.0), (12.0, 24.0), (24.0, None)),
        use_euclidean_distance=True,
        height_cap=height_cap_value,   # 只统计 z ≤ cap 的点
    )
    # clean_res['max_height_m'] 就是你要的最高点（米）


    # # 2) 直接按 (weather, severity) 全量出表
    # table = compute_range_bin_table_by_buckets(
    #     csv_path=csv_path,
    #     raw_lidar_root=raw_lidar_root,
    #     split=SPLIT,
    #     bucket_fields=("weather", "severity"),
    #     range_bins=((0.0, 12.0), (12.0, 24.0), (24.0, None)),
    #     use_euclidean_distance=True,
    # )