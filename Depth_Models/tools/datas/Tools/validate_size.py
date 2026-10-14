import os
import sys
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义图片扩展名
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def get_single_image_size(file_path):
    """
    读取单张图片大小的独立函数，用于多线程调用
    """
    try:
        with Image.open(file_path) as img:
            return img.size  # 返回 (width, height)
    except Exception:
        return None

def get_folder_resolutions_parallel(directory, max_workers=16):
    """
    使用多线程获取目录下所有图片的分辨率
    """
    image_files = []
    resolution_stats = {}
    
    # 1. 先快速收集所有图片路径 (这一步必须是串行的，但通常很快)
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(dirpath, f))

    if not image_files:
        return {}

    # 2. 并行读取图片头信息
    # max_workers 可以根据电脑性能调整，网络驱动器建议设高一点(如 32 或 64)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(get_single_image_size, fp): fp for fp in image_files}
        
        # 获取结果
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                if result in resolution_stats:
                    resolution_stats[result] += 1
                else:
                    resolution_stats[result] = 1
                    
    return resolution_stats

def list_subfolder_resolutions(path):
    path = path.strip()
    if not os.path.exists(path):
        print(f"错误: 路径 '{path}' 不存在。")
        return

    print(f"正在分析路径: {path}")
    print("正在使用多线程加速读取，请稍候...\n")

    # 表头
    print(f"{'Folder Name':<40} | {'Resolution (WxH) [Count]':<40}")
    print("-" * 90)

    subfolders = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    subfolders.append(entry.name)
    except PermissionError:
        print(f"错误: 没有权限读取路径 {path}")
        return

    subfolders.sort()

    total_start_time = time.time()

    for folder_name in subfolders:
        full_sub_path = os.path.join(path, folder_name)
        
        # 调用并行处理函数
        # 如果是在网络位置运行，建议把 max_workers 调大到 32 甚至 64
        stats = get_folder_resolutions_parallel(full_sub_path, max_workers=32)
        
        if not stats:
            res_str = "No images found"
        else:
            # 排序并格式化
            sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)
            res_list = [f"{w}x{h} [{cnt}]" for (w, h), cnt in sorted_stats]
            res_str = ", ".join(res_list)

        print(f"{folder_name:<40} | {res_str}")

    print("-" * 90)
    print(f"分析完成，耗时: {time.time() - total_start_time:.2f} 秒")

if __name__ == "__main__":
    # --- 你的目标路径 ---
    target_path = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC/data_depth_velodyne/train/2011_09_28_drive_0113_sync/proj_depth/velodyne_raw"
    
    list_subfolder_resolutions(target_path)