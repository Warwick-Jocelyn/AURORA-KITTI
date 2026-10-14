import os
import numpy as np
from PIL import Image

def convert_mixed_depth_images(input_dir, output_dir):
    """
    处理混合类型的深度图：
    - 16bit灰度图：保持不变
    - 24bit RGB图：取三通道平均，转换为16bit灰度保存
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.tiff', '.bmp')):
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        with Image.open(in_path) as img:
            arr = np.array(img)
            dtype = arr.dtype
            channels = arr.shape[-1] if arr.ndim == 3 else 1

            print(f"🔍 {filename}: dtype={dtype}, channels={channels}, shape={arr.shape}")

            # ---- 处理 16-bit 单通道图 ----
            if dtype == np.uint16 and channels == 1:
                print("➡️ 16-bit 单通道图，直接复制。")
                img.save(out_path)

            # ---- 处理 24-bit (8-bit × 3通道) 图 ----
            elif dtype == np.uint8 and channels == 3:
                # 检查三个通道是否相近
                diff_12 = np.abs(arr[...,0].astype(np.int16) - arr[...,1].astype(np.int16))
                diff_13 = np.abs(arr[...,0].astype(np.int16) - arr[...,2].astype(np.int16))
                max_diff = max(diff_12.max(), diff_13.max())

                if max_diff < 2:  # 通道值几乎一致（0或1像素差）
                    print(f"➡️ 通道近似相同 (max_diff={max_diff})，取平均转换为16-bit。")
                    gray = np.mean(arr, axis=-1)  # 平均后仍为uint8范围0~255
                    gray16 = (gray.astype(np.uint16) * 256)  # 转换为16-bit范围
                    Image.fromarray(gray16, mode='I;16').save(out_path)
                else:
                    print(f"⚠️ 通道差异较大 (max_diff={max_diff})，保留原图。")
                    img.save(out_path)

            # ---- 其他格式 ----
            else:
                print(f"⚠️ 未识别格式 (dtype={dtype}, channels={channels})，直接保存。")
                img.save(out_path)

        print(f"✅ 已处理并保存到: {out_path}\n")


if __name__ == "__main__":
    input_dir = "/path/to/aurora-root/Code/Depth_visualisation/Depth_anything_input"
    output_dir = "/path/to/aurora-root/Code/Depth_visualisation/depth_output"
    convert_mixed_depth_images(input_dir, output_dir)