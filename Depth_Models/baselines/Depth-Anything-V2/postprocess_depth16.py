# postprocess_depth16.py
import os
import sys
from PIL import Image
import numpy as np

def ensure_16bit_depth(img_path):
    """
    把一张图转成16bit深度图:
    - 如果已经是16bit (mode='I;16' / dtype uint16)，保持不变
    - 如果是8bit灰度，线性拉伸到0-65535再保存为16bit
    - 如果是float深度(常见是0~1 或 实际深度米值的float32)，会归一化后转16bit
    """
    img = Image.open(img_path)

    # 尝试拿到底层数组
    arr = np.array(img)

    # 情况1：已经是 uint16，就直接跳过
    if arr.dtype == np.uint16:
        # double check: 确保保存格式也是16bit PNG（有些库会偷懒存成8bit）
        # 我们还是重新save一次保证一致性
        Image.fromarray(arr).save(img_path, mode='I;16')
        return False  # False表示没做强制提升（只是可能重存）

    # 情况2：8bit图 (uint8)
    if arr.dtype == np.uint8:
        # 拉伸到0-65535
        arr16 = (arr.astype(np.uint16) * 257)  # 255 * 257 = 65535，线性放大
        Image.fromarray(arr16).save(img_path, mode='I;16')
        return True  # True表示做了转换

    # 情况3：float图，比如float32/float16
    if np.issubdtype(arr.dtype, np.floating):
        # 做一个安全归一化：
        #   - 如果数值范围在[0,1]，那就直接 *65535
        #   - 否则我们做min-max归一化到[0,1]后再乘65535
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if vmax == vmin:
            # 全同值，直接全零
            arr_norm = np.zeros_like(arr, dtype=np.float32)
        else:
            # 判断是不是已经在[0,1]
            if vmin >= 0.0 and vmax <= 1.0:
                arr_norm = arr
            else:
                arr_norm = (arr - vmin) / (vmax - vmin)

        arr16 = np.clip(arr_norm * 65535.0, 0, 65535).astype(np.uint16)
        Image.fromarray(arr16).save(img_path, mode='I;16')
        return True

    # 其他类型（比如RGB三通道之类）
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        # 如果意外输出成RGB，就先转灰度再升16bit
        gray8 = np.array(img.convert("L"))
        arr16 = (gray8.astype(np.uint16) * 257)
        Image.fromarray(arr16).save(img_path, mode='I;16')
        return True

    # fallback：直接强转uint16（可能信息有损，但至少满足16bit要求）
    arr16 = arr.astype(np.uint16)
    Image.fromarray(arr16).save(img_path, mode='I;16')
    return True


def process_dir(outdir):
    changed = 0
    total = 0
    for name in os.listdir(outdir):
        if not name.lower().endswith((".png", ".tiff", ".tif")):
            continue
        total += 1
        path = os.path.join(outdir, name)
        if ensure_16bit_depth(path):
            changed += 1

    print(f"[完成] 共检查 {total} 张图，其中 {changed} 张被转换成16bit。")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python postprocess_depth16.py /path/to/outdir")
        sys.exit(1)

    outdir = sys.argv[1]
    process_dir(outdir)
