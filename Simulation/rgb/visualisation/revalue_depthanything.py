import os
import numpy as np
from PIL import Image

def process_depth_anything_folder(input_dir, output_dir, max_m=80.0):
    """
    处理 Depth Anything 生成的 disparity maps：
    - 将像素值归一化到 [0, max_m]
    - 反转为 (max_m - value)
    - 保存为 16-bit 单通道 PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.tiff', '.bmp')):
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        with Image.open(in_path) as img:
            arr = np.array(img, dtype=np.uint16)

            if arr.ndim != 2:
                raise ValueError(f"输入图像必须为单通道16bit灰度: {filename} (shape={arr.shape})")

            # 1️⃣ 将像素映射到 0~1
            arr_float = arr.astype(np.float32)
            arr_min, arr_max = np.min(arr_float), np.max(arr_float)
            if arr_max == arr_min:
                print(f"⚠️ 图像 {filename} 所有像素相同，跳过。")
                continue

            arr_norm = (arr_float - arr_min) / (arr_max - arr_min)

            # 2️⃣ 映射到 [0, max_m]
            depth_m = arr_norm * max_m

            # 3️⃣ 反转深度（让近的变远）
            depth_inverted = max_m - depth_m

            # 4️⃣ 转回 16-bit PNG格式（米 × 256）
            depth_uint16 = np.clip(depth_inverted * 256.0, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_uint16, mode="I;16").save(out_path)

            print(f"✅ 已处理: {filename} -> {out_path}")

    print("🎉 所有图像处理完成！")


if __name__ == "__main__":
    input_dir = "/path/to/aurora-root/Code/Depth_visualisation/Depth_anything_input"
    output_dir = "/path/to/aurora-root/Code/Depth_visualisation/depth_output"
    process_depth_anything_folder(input_dir, output_dir, max_m=80.0)
