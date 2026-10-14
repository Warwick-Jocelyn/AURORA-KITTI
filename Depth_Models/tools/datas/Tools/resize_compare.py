import os
import cv2
import numpy as np
from pathlib import Path

# --- 核心处理逻辑 (对应你提供的 resize_png_folder_cv2 中的核心操作) ---
def process_image_in_memory(img_path, target_size=(1242, 375)):
    """
    读取图像并使用 cv2.INTER_AREA 进行直接缩放 (无 padding)。
    """
    # 1. 读取图像
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法找到或读取图像: {img_path}")
    
    # 获取原始分辨率 (用于后续计算)
    h_origin, w_origin = img.shape[:2]
    original_pixel_count = h_origin * w_origin

    # 2. 执行缩放 (核心逻辑)
    # 对应代码: resized = cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 获取新分辨率
    h_new, w_new = resized_img.shape[:2]
    new_pixel_count = h_new * w_new

    return resized_img, original_pixel_count, new_pixel_count

# --- 对比与计算函数 ---
def verify_and_calculate(img_path_a, img_path_b, target_size=(1242, 375)):
    print(f"🔹 正在对比两张图片...")
    print(f"   A: {img_path_a}")
    print(f"   B: {img_path_b}")
    print("-" * 40)

    # 1. 分别处理两张图片
    try:
        res_a, orig_pixels_a, new_pixels_a = process_image_in_memory(img_path_a, target_size)
        res_b, orig_pixels_b, new_pixels_b = process_image_in_memory(img_path_b, target_size)
    except Exception as e:
        print(f"❌ 出错: {e}")
        return

    # --- 计算指标 1: Resize后两张图之间的差异像素数量 ---
    # 如果两张图原本是一样的，这里应该是 0
    diff_img = cv2.absdiff(res_a, res_b)
    
    # 将差异转为灰度，只要某个像素 RGB 任何一个通道不同，就不为0
    if len(diff_img.shape) == 3:
        diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff_img
        
    # 计算非零像素个数（即不一样的像素点数量）
    diff_pixel_count = np.count_nonzero(diff_gray)

    # --- 计算指标 2: Resize前后分辨率带来的像素数量差异 ---
    # (原图总像素 - 缩放后总像素)
    resolution_diff_pixels = abs(orig_pixels_a - new_pixels_a)

    # --- 输出结果 ---
    print(f"📊【分析报告】")
    
    print(f"\n1. 图像一致性检查 (Resize后 A vs B):")
    if diff_pixel_count == 0:
        print(f"   ✅ 结果: 完全一致。")
    else:
        print(f"   ❌ 结果: 不一致！")
    print(f"   👉 不一样的像素数量: {diff_pixel_count} (像素点)")

    print(f"\n2. 分辨率变化统计 (原图 vs Resize后):")
    print(f"   原图像素总数: {orig_pixels_a} (例如 {res_a.shape[1]}x{res_a.shape[0]} 缩放前)")
    print(f"   目标像素总数: {new_pixels_a} ({target_size[0]}x{target_size[1]})")
    print(f"   👉 像素数量差异 (减少/增加的像素): {resolution_diff_pixels}")

    print(f"\n3. 数据一致性对比:")
    if diff_pixel_count == resolution_diff_pixels:
        print(f"   ⚠️ 巧合: 两图差异像素数 刚好等于 分辨率缩减像素数 (这通常极少发生，除非其中一张是纯黑等特殊情况)。")
    else:
        print(f"   ℹ️ 正常: '两图差异数量' 与 '分辨率缩减数量' 不一致 (这是两个不同的概念)。")
    
    # 保存差异图以便观察
    if diff_pixel_count > 0:
        cv2.imwrite("diff_check.png", diff_img * 10) # 放大差异亮度
        print(f"\n   (已保存差异图像到 diff_check.png)")

if __name__ == "__main__":
    # 请替换为你要验证的两张实际图片路径
    # 假设这里我们验证两张图，或者验证“修改前”和“修改后”的图
    image_a = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/image_02/data/0000000055.png"
    image_b = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/image_02/data/0000000055.png"
    
    # 确保文件存在再运行 (这里仅作演示，实际运行请填入真实路径)
    if os.path.exists(image_a) and os.path.exists(image_b):
        verify_and_calculate(image_a, image_b, target_size=(1242, 375))
    else:
        print("请在代码中修改 IMG_A 和 IMG_B 为真实的图片路径。")


