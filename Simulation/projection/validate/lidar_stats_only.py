import os
import csv
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------------------------
# 工具函数
# ---------------------------

def check_dir_has_bins(dir_path: str) -> List[str]:
    if not dir_path:
        return []
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".bin")])
    if not files:
        raise FileNotFoundError(f"目录中未找到 .bin 文件: {dir_path}")
    return files

def normalize_angle_deg(a: np.ndarray) -> np.ndarray:
    """将角度归一化到 [-180, 180] 区间"""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def sanitize_fov(min_deg: float, max_deg: float, kind: str) -> Tuple[float, float]:
    """修正 FOV 边界与范围，返回 [min, max]（度）"""
    if min_deg is None or max_deg is None:
        raise ValueError(f"{kind} FOV 需要提供 min 和 max")
    # 归一化到 [-180, 180]
    def norm(x): return (x + 180.0) % 360.0 - 180.0
    mn, mx = norm(min_deg), norm(max_deg)
    # 若给反了，自动交换
    if mn > mx:
        mn, mx = mx, mn
    # 垂直 FOV 实际物理范围通常不超过 [-90, 90]，做个温和警告（不强改）
    if kind.lower().startswith("vertical") and (mn < -90 or mx > 90):
        print(f"⚠️ 垂直FOV超出常见范围: [{mn:.2f}, {mx:.2f}] 度（仅提示）")
    return mn, mx

def safe_read_points(file_path: str) -> Optional[np.ndarray]:
    """读取 Nx4 float32 点云，出现异常时返回 None 并打印告警"""
    try:
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size == 0:
            print(f"⚠️ 空文件，跳过: {file_path}")
            return None
        if raw.size % 4 != 0:
            print(f"⚠️ 文件长度不是4的倍数，跳过: {file_path}")
            return None
        pts = raw.reshape(-1, 4)
        # 过滤 NaN / Inf
        if not np.isfinite(pts).all():
            mask = np.isfinite(pts).all(axis=1)
            n_bad = (~mask).sum()
            pts = pts[mask]
            print(f"⚠️ 检测到 {n_bad} 个含 NaN/Inf 的点，已过滤: {file_path}")
        if pts.shape[0] == 0:
            print(f"⚠️ 过滤后无有效点，跳过: {file_path}")
            return None
        return pts
    except Exception as e:
        print(f"❌ 读取失败，跳过 {file_path}: {e}")
        return None

def per_frame_stats(pts_xyz: np.ndarray,
                    h_min: float, h_max: float,
                    v_min: float, v_max: float) -> Tuple[int, float, int]:
    """
    计算单帧统计：
    - points: 点数
    - mean_point_distance: 平均点距
    - points_in_image: 视场内点数（h/v FOV 闭区间）
    """
    x, y, z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
    ranges = np.linalg.norm(pts_xyz, axis=1)
    mean_range = float(ranges.mean()) if ranges.size > 0 else 0.0

    yaw = np.degrees(np.arctan2(y, x))             # [-180, 180]
    pitch = np.degrees(np.arctan2(z, np.hypot(x, y)))  # [-90, 90]
    yaw = normalize_angle_deg(yaw)

    in_h = (yaw >= h_min) & (yaw <= h_max)
    in_v = (pitch >= v_min) & (pitch <= v_max)
    in_fov = in_h & in_v
    n_in_fov = int(np.count_nonzero(in_fov))

    return int(pts_xyz.shape[0]), mean_range, n_in_fov

# ---------------------------
# 目录级别统计
# ---------------------------

def process_one_dir(dir_path: str,
                    out_csv: str,
                    h_fov_min: float, h_fov_max: float,
                    v_fov_min: float, v_fov_max: float) -> Dict[str, float]:
    """对单个目录逐帧统计，并写出明细 & 汇总"""
    files = check_dir_has_bins(dir_path)
    h_min, h_max = sanitize_fov(h_fov_min, h_fov_max, "Horizontal")
    v_min, v_max = sanitize_fov(v_fov_min, v_fov_max, "Vertical")

    rows = []
    sum_points = 0
    sum_mean_range = 0.0
    sum_in_img = 0
    n_frames = 0

    print(f"\n📂 开始统计目录: {dir_path}")
    print(f"   使用 FOV: H=[{h_min:.2f}, {h_max:.2f}]°, V=[{v_min:.2f}, {v_max:.2f}]°")

    for f in files:
        fp = os.path.join(dir_path, f)
        pts = safe_read_points(fp)
        if pts is None:
            continue
        xyz = pts[:, :3]
        points, mean_range, in_img = per_frame_stats(xyz, h_min, h_max, v_min, v_max)

        rows.append({
            "filename": f,
            "points": points,
            "mean_point_distance": mean_range,
            "points_in_image": in_img,
        })

        sum_points += points
        sum_mean_range += mean_range
        sum_in_img += in_img
        n_frames += 1

        print(f"{f}: points={points}, mean_range={mean_range:.3f}, in_image={in_img}")

    if n_frames == 0:
        raise RuntimeError(f"该目录没有成功统计的帧（可能文件均损坏或为空）: {dir_path}")

    avg_points = sum_points / n_frames
    avg_mean_range = sum_mean_range / n_frames
    avg_in_img = sum_in_img / n_frames

    # 写明细
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "points", "mean_point_distance", "points_in_image"])
        writer.writeheader()
        writer.writerows(rows)

    # 写汇总
    summary_csv = os.path.splitext(out_csv)[0] + "_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["Average of points", avg_points])
        w.writerow(["Average point distance", avg_mean_range])
        w.writerow(["Average of points in image", avg_in_img])

    print(f"\n✅ 明细保存: {out_csv}")
    print(f"✅ 汇总保存: {summary_csv}")
    return {
        "avg_points": avg_points,
        "avg_mean_point_distance": avg_mean_range,
        "avg_points_in_image": avg_in_img,
        "frames": n_frames
    }

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LiDAR 目录统计（无 drop rate）")
    p.add_argument("--dir_a", type=str, required=True,
                   help="要统计的目录 A（必填）")
    p.add_argument("--dir_b", type=str, required=False, default="",
                   help="可选：目录 B（若提供则会分别统计 A 与 B）")
    p.add_argument("--h_fov_min", type=float, default=-180.0,
                   help="水平视场最小角度（度）")
    p.add_argument("--h_fov_max", type=float, default=180.0,
                   help="水平视场最大角度（度）")
    p.add_argument("--v_fov_min", type=float, default=-24.9,
                   help="垂直视场最小角度（度）")
    p.add_argument("--v_fov_max", type=float, default=3.0,
                   help="垂直视场最大角度（度）")
    p.add_argument("--out_csv_a", type=str, default="stats_A.csv",
                   help="目录 A 明细输出路径")
    p.add_argument("--out_csv_b", type=str, default="stats_B.csv",
                   help="目录 B 明细输出路径（仅在提供 dir_b 时使用）")
    return p.parse_args()

def main():
    args = parse_args()

    # 只统计 A
    summary_a = process_one_dir(
        dir_path=args.dir_a,
        out_csv=args.out_csv_a,
        h_fov_min=args.h_fov_min, h_fov_max=args.h_fov_max,
        v_fov_min=args.v_fov_min, v_fov_max=args.v_fov_max
    )

    # 若提供 B，则额外统计 B（互不做比例/差值）
    if args.dir_b:
        summary_b = process_one_dir(
            dir_path=args.dir_b,
            out_csv=args.out_csv_b,
            h_fov_min=args.h_fov_min, h_fov_max=args.h_fov_max,
            v_fov_min=args.v_fov_min, v_fov_max=args.v_fov_max
        )
        print("\n====== Dataset-level Averages ======")
        print(f"A: frames={summary_a['frames']}, "
              f"avg_points={summary_a['avg_points']:.2f}, "
              f"avg_mean_distance={summary_a['avg_mean_point_distance']:.4f}, "
              f"avg_points_in_image={summary_a['avg_points_in_image']:.2f}")
        print(f"B: frames={summary_b['frames']}, "
              f"avg_points={summary_b['avg_points']:.2f}, "
              f"avg_mean_distance={summary_b['avg_mean_point_distance']:.4f}, "
              f"avg_points_in_image={summary_b['avg_points_in_image']:.2f}")
    else:
        print("\n====== Dataset-level Averages (A only) ======")
        print(f"A: frames={summary_a['frames']}, "
              f"avg_points={summary_a['avg_points']:.2f}, "
              f"avg_mean_distance={summary_a['avg_mean_point_distance']:.4f}, "
              f"avg_points_in_image={summary_a['avg_points_in_image']:.2f}")

if __name__ == "__main__":
    main()





# python lidar_stats_only.py \
#   --dir_a "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/raw_lidar_clear/test" \
#   --dir_b "/path/to/aurora-root/PhD_Project_2025/lidar_compound_noise_models_original/raw_lidar_simulated_Fog/test" \
#   --h_fov_min -180 --h_fov_max 180 \
#   --v_fov_min -24.9 --v_fov_max 3.0 \
#   --out_csv_a stats_A.csv --out_csv_b stats_B.csv
