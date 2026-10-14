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
):
    """
    读取 CSV，按 weather / severity 过滤后，
    统计所选样本中三个距离段的点数比例（跨样本整体统计）。

    返回：
      result(dict): {
        'total_points': int,
        'bin_counts': [c0_20, c20_40, c40p],
        'bin_props':  [p0_20, p20_40, p40p],
        'skipped':    int,   # 缺失 .bin 的样本数
        'num_samples':int    # 有效样本数（成功读取 .bin）
      }
    """
    df = pd.read_csv(csv_path)

    # weather 过滤
    if weather:
        df = df[df['weather'] == weather]

    # severity 过滤（支持单个、多选、逗号字符串）
    sev_list = _to_list_or_none(severity)
    if sev_list:
        df = df[df['severity'].isin(sev_list)]

    print(f"🗂️ Split={split} | weather={weather} | severity={sev_list} | candidates={len(df)}")

    total_counts = np.zeros(len(range_bins), dtype=np.int64)
    total_points = 0
    skipped = 0
    valid_samples = 0

    for _, row in df.iterrows():
        name = row['name']
        bin_path = build_bin_path_from_name(name, raw_lidar_root, split)
        if not bin_path or not os.path.isfile(bin_path):
            # 缺失 .bin：跳过
            skipped += 1
            continue

        try:
            pts = read_kitti_bin_points(bin_path)  # (N,4)
        except Exception as e:
            print(f"[Skip] {bin_path}: {e}")
            skipped += 1
            continue

        # 计算距离：默认欧氏距离；如需前向距离，用 x 轴即可（见下一行注释）
        if use_euclidean_distance:
            dists = np.linalg.norm(pts[:, :3], axis=1)  # sqrt(x^2+y^2+z^2)
        else:
            dists = np.abs(pts[:, 0])  # 仅使用 x（前向）作为距离代理

        # 桶计数
        bin_counts = np.zeros(len(range_bins), dtype=np.int64)
        for i, (lo, hi) in enumerate(range_bins):
            if hi is None:
                mask = dists >= lo
            else:
                mask = (dists >= lo) & (dists < hi)
            bin_counts[i] = int(mask.sum())

        total_counts += bin_counts
        total_points += int(dists.size)
        valid_samples += 1

    if total_points == 0:
        print("❗没有有效点云可统计（可能 .bin 缺失或过滤过严）")
        return {
            'total_points': 0,
            'bin_counts': [0]*len(range_bins),
            'bin_props': [0.0]*len(range_bins),
            'skipped': skipped,
            'num_samples': valid_samples,
        }

    bin_props = (total_counts / total_points).tolist()
    result = {
        'total_points': int(total_points),
        'bin_counts': total_counts.tolist(),
        'bin_props': bin_props,
        'skipped': skipped,
        'num_samples': valid_samples,
    }

    # 打印简报
    labels = [f"[{lo},{'∞' if hi is None else hi}) m" for (lo, hi) in range_bins]
    print(f"✅ 有效样本: {valid_samples} | 跳过: {skipped}")
    for lab, c, p in zip(labels, result['bin_counts'], result['bin_props']):
        print(f"  {lab:>10}: {c:>10d} pts | {p*100:6.2f}%")
    print(f"  {'TOTAL':>10}: {result['total_points']:>10d} pts")

    return result

def compute_range_bin_table_by_buckets(
    csv_path: str,
    raw_lidar_root: str,
    split: str,
    bucket_fields=('weather', 'severity'),
    range_bins=((0.0, 20.0), (20.0, 40.0), (40.0, None)),
    use_euclidean_distance: bool = True,
) -> pd.DataFrame:
    """
    按 (weather, severity) 分桶统计三段距离占比，返回一个 DataFrame 便于保存/画表。
    """
    df = pd.read_csv(csv_path)
    # 仅保留当前 split 的样本（若 CSV 里混有多个 split，可在 CSV 里加列并在此处过滤）
    # 如果你的 CSV 已经是每个 split 一个文件，这里不需要额外过滤。

    # 逐桶计算
    rows = []
    # 若 CSV 很大，这里的 groupby 会非常高效
    grouped = df.groupby(list(bucket_fields), dropna=False)
    for bucket_vals, subdf in grouped:
        # 允许 bucket_vals 是标量或元组
        if not isinstance(bucket_vals, tuple):
            bucket_vals = (bucket_vals,)

        # 将该桶写入临时 CSV（或直接复用 DataFrame 行）
        # 这里直接遍历 subdf 逐样本处理，逻辑与上面函数相同
        total_counts = np.zeros(len(range_bins), dtype=np.int64)
        total_points = 0
        skipped = 0
        valid_samples = 0

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

            if use_euclidean_distance:
                dists = np.linalg.norm(pts[:, :3], axis=1)
            else:
                dists = np.abs(pts[:, 0])

            bin_counts = np.zeros(len(range_bins), dtype=np.int64)
            for i, (lo, hi) in enumerate(range_bins):
                if hi is None:
                    mask = dists >= lo
                else:
                    mask = (dists >= lo) & (dists < hi)
                bin_counts[i] = int(mask.sum())

            total_counts += bin_counts
            total_points += int(dists.size)
            valid_samples += 1

        if total_points == 0:
            props = [0.0]*len(range_bins)
        else:
            props = (total_counts / total_points).tolist()

        row_dict = {k: v for k, v in zip(bucket_fields, bucket_vals)}
        # 计数与比例展开
        for i, (lo, hi) in enumerate(range_bins):
            lab = f"[{lo},{'inf' if hi is None else hi})m"
            row_dict[f"count{lab}"] = int(total_counts[i])
            row_dict[f"prop{lab}"]  = float(props[i])

        row_dict["total_points"] = int(total_points)
        row_dict["num_samples"]  = int(valid_samples)
        row_dict["skipped"]      = int(skipped)
        rows.append(row_dict)

    out_df = pd.DataFrame(rows).sort_values(list(bucket_fields)).reset_index(drop=True)
    return out_df

# ======== 使用示例 ========
if __name__ == "__main__":
    SPLIT = "train"
    base  = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    csv_path       = f"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/{SPLIT}.csv"
    raw_lidar_root = f"{base}/raw_lidar"

    # 1) 针对单一筛选（例如：Clear + severity in ['None']）
    res = compute_range_bin_proportions(
        csv_path=csv_path,
        raw_lidar_root=raw_lidar_root,
        split=SPLIT,
        weather="Clear",
        severity=[" None"],           # 或 "None" / "None,light"
        range_bins=((0.0, 10.0), (10.0, 20.0), (20.0, None)),
        use_euclidean_distance=True,
    )

    # 2) 直接按 (weather, severity) 全量出表
    table = compute_range_bin_table_by_buckets(
        csv_path=csv_path,
        raw_lidar_root=raw_lidar_root,
        split=SPLIT,
        bucket_fields=("weather", "severity"),
        range_bins=((0.0, 12.0), (12.0, 24.0), (24.0, None)),
        use_euclidean_distance=True,
    )
    # 保存统计表（可选）
    # out_csv = f"./range_bin_proportions_{SPLIT}.csv"
    # table.to_csv(out_csv, index=False)
    # print(f"💾 Saved table to: {out_csv}")


# 🗂️ Split=val | weather=Clear | severity=[' None'] | candidates=823
# ✅ 有效样本: 823 | 跳过: 0
#   [0.0,20.0) m:   79816930 pts |  81.27%
#   [20.0,40.0) m:   13997754 pts |  14.25%
#   [40.0,∞) m:    4401339 pts |   4.48%
#        TOTAL:   98216023 pts
# 💾 Saved table to: ./range_bin_proportions_val.csv