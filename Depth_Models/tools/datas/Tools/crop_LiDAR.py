import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np  # 引入 numpy 进行统计计算
from tqdm import tqdm

# ================= 配置区域 =================
TARGET_W = 1216
TARGET_H = 352

# 支持的格式
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
# ===========================================

def compute_crop_box(w, h, target_w, target_h):
    """
    计算 Center-Bottom Crop 的坐标
    """
    if w < target_w or h < target_h:
        return None

    # 高度：保留底部
    top = h - target_h
    bottom = h
    
    # 宽度：居中
    to_crop_w = w - target_w
    left = to_crop_w // 2
    right_extra = to_crop_w - left 
    right = w - right_extra
    
    # Clamp (防止越界)
    left = max(0, min(left, w - target_w))
    right = max(left + target_w, min(w, right))
    top = max(0, min(top, h - target_h))
    bottom = max(top + target_h, min(h, bottom))
    
    return (left, top, right, bottom)

def calculate_stats(img_pil):
    """
    将 PIL 图片转为 numpy 数组并计算最大值和中位数
    """
    arr = np.array(img_pil)
    return {
        "max": np.max(arr),
        "median": np.median(arr)
    }

def process_images(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    
    if not all_files:
        print(f"❌ 未找到图片文件: {input_dir}")
        return

    print(f"📂 输入: {input_dir}")
    print(f"📂 输出: {output_dir}")
    print(f"✂️  Target: {TARGET_W}x{TARGET_H}")
    print("-" * 80)

    stats_report = []

    for file_path in tqdm(all_files, desc="Processing"):
        try:
            with Image.open(file_path) as img:
                orig_w, orig_h = img.size
                
                # --- 1. Validate: Crop 前统计 ---
                stats_pre = calculate_stats(img)

                # --- 2. Crop 操作 ---
                box = compute_crop_box(orig_w, orig_h, TARGET_W, TARGET_H)
                
                if box:
                    cropped_img = img.crop(box)
                    final_w, final_h = cropped_img.size
                    
                    # --- 3. Validate: Crop 后统计 ---
                    stats_post = calculate_stats(cropped_img)
                    
                    # 保存
                    save_path = output_path / file_path.name
                    # 保持原始格式和深度 (特别是 16-bit png)
                    cropped_img.save(save_path)
                    
                    stats_report.append({
                        "name": file_path.name,
                        "orig_size": f"{orig_w}x{orig_h}",
                        "crop_size": f"{final_w}x{final_h}",
                        "pre_max": stats_pre['max'],
                        "post_max": stats_post['max'],
                        "pre_med": stats_pre['median'],
                        "post_med": stats_post['median'],
                        "status": "OK"
                    })
                else:
                    stats_report.append({
                        "name": file_path.name,
                        "orig_size": f"{orig_w}x{orig_h}",
                        "crop_size": "-",
                        "status": "Skipped (Small)"
                    })

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # --- 打印 Validate 报表 ---
    print("\n" + "="*110)
    # 表头格式化
    header = (f"{'File Name':<30} | {'Size (Pre->Post)':<20} | "
              f"{'Max (Pre->Post)':<20} | {'Median (Pre->Post)':<20}")
    print(header)
    print("-" * 110)
    
    for item in stats_report:
        if item["status"] != "OK":
            print(f"{item['name']:<30} | {item['orig_size']:<20} | {item['status']}")
            continue

        # 格式化数值显示
        size_str = f"{item['orig_size']} -> {item['crop_size']}"
        
        # 使用 :.1f 控制小数位数，让显示更整齐
        max_str = f"{item['pre_max']:.1f} -> {item['post_max']:.1f}"
        med_str = f"{item['pre_med']:.1f} -> {item['post_med']:.1f}"
        
        # 检查数值是否有剧烈变化 (可选: 给个警告标记)
        # 例如: 如果最大值变了，说明原来的最大值在被切掉的区域
        flag = ""
        if item['pre_max'] != item['post_max']:
            flag = "(*)" # 标记最大值发生了改变
            
        print(f"{item['name']:<30} | {size_str:<20} | {max_str:<20} {flag}| {med_str:<20}")
            
    print("="*110)
    print("(*) 表示 Crop 前后最大值发生了改变 (说明极值点位于被裁剪区域)")
    print("✅ 处理完成！")

if __name__ == "__main__":
    # python crop.py ./velodyne_raw_original ./velodyne_raw
    if len(sys.argv) < 3:
        print("Usage: python crop_tool_validate.py <input_folder> <output_folder>")
    else:
        process_images(sys.argv[1], sys.argv[2])


# python crop_LiDAR.py /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST/velodyne_raw_not_crop /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_TEST/velodyne_raw