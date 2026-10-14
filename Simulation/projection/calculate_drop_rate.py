import os
import numpy as np

def count_points_in_bin(file_path):
    """读取 .bin 文件并返回点云数量"""
    # 每个点是4个float32(x, y, z, intensity)，每个float32是4字节
    points = np.fromfile(file_path, dtype=np.float32)
    return points.reshape(-1, 4).shape[0]

def calculate_drop_rate(clean_lidar_path, degraded_path):
    clean_files = sorted([f for f in os.listdir(clean_lidar_path) if f.endswith(".bin")])
    degraded_files = sorted([f for f in os.listdir(degraded_path) if f.endswith(".bin")])

    # 确保两个文件夹中的文件数量一致
    if len(clean_files) != len(degraded_files):
        print(f"⚠️ 文件数量不一致: clean={len(clean_files)} vs degraded={len(degraded_files)}")

    results = []
    for clean_file, degraded_file in zip(clean_files, degraded_files):
        clean_fp = os.path.join(clean_lidar_path, clean_file)
        degraded_fp = os.path.join(degraded_path, degraded_file)

        A = count_points_in_bin(clean_fp)
        B = count_points_in_bin(degraded_fp)
        drop_rate = (A - B) / A * 100

        results.append((clean_file, A, B, drop_rate))
        print(f"{clean_file} -> A={A}, B={B}, drop_rate={drop_rate:.2f}%")

    return results

if __name__ == "__main__":
    clean_lidar_path = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/raw_lidar_clear/test"
    degraded_path = "/path/to/aurora-root/PhD_Project_2025/lidar_compound_noise_models_original/raw_lidar_simulated_Fog/test"

    results = calculate_drop_rate(clean_lidar_path, degraded_path)

    # 如果想导出到CSV
    import csv
    with open("drop_rate_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "A_clean_points", "B_degraded_points", "drop_rate(%)"])
        writer.writerows(results)
    print("\n✅ 结果已保存到 drop_rate_results.csv")
