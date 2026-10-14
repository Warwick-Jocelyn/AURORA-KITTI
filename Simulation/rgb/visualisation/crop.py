import os
import shutil
from PIL import Image

INPUT_ROOT = "/path/to/aurora-root/Code/Depth_visualisation/Inputs_DD3M"
OUTPUT_ROOT = "/path/to/aurora-root/Code/Depth_visualisation/Inputs_DD3M_cropped"

# 常见图片后缀（可按需增减）
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TARGET_W, TARGET_H_IN = 1216, 352
TARGET_H_OUT = 256

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def crop_if_needed(in_path: str, out_path: str) -> str:
    """
    如果图片尺寸是 1216x352，则从顶部裁掉，使其变成 1216x256。
    其他尺寸：原样复制（也可改成跳过）。
    返回处理状态字符串。
    """
    try:
        with Image.open(in_path) as img:
            w, h = img.size
            if (w, h) == (TARGET_W, TARGET_H_IN):
                # 从顶部裁掉 96 像素：保留 y=96 到 y=352（共 256）
                top_cut = TARGET_H_IN - TARGET_H_OUT  # 96
                cropped = img.crop((0, top_cut, TARGET_W, TARGET_H_IN))

                ensure_dir(os.path.dirname(out_path))
                # 尽量保留原图格式
                cropped.save(out_path)
                return "cropped"
            else:
                # 其他尺寸：原样复制
                ensure_dir(os.path.dirname(out_path))
                shutil.copy2(in_path, out_path)
                return f"copied({w}x{h})"
    except Exception as e:
        return f"error: {e}"

def main():
    ensure_dir(OUTPUT_ROOT)

    total = 0
    cropped_cnt = 0
    copied_cnt = 0
    err_cnt = 0

    for root, _, files in os.walk(INPUT_ROOT):
        rel_dir = os.path.relpath(root, INPUT_ROOT)  # 相对路径
        out_dir = os.path.join(OUTPUT_ROOT, rel_dir) if rel_dir != "." else OUTPUT_ROOT

        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS:
                continue

            total += 1
            in_path = os.path.join(root, fn)
            out_path = os.path.join(out_dir, fn)

            status = crop_if_needed(in_path, out_path)
            if status == "cropped":
                cropped_cnt += 1
            elif status.startswith("copied"):
                copied_cnt += 1
            else:
                err_cnt += 1
                print(f"[FAIL] {in_path} -> {out_path} | {status}")

    print("Done.")
    print(f"Total images processed: {total}")
    print(f"Cropped (1216x352 -> 1216x256): {cropped_cnt}")
    print(f"Copied (other sizes): {copied_cnt}")
    print(f"Errors: {err_cnt}")

if __name__ == "__main__":
    main()
