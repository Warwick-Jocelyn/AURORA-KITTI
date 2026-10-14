import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 允许的输入图像扩展名
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def run_single_task(cmd, data_path):
    """执行单个 run.py 命令"""
    try:
        print(f"🚀 Start: {data_path}")
        subprocess.run(cmd, check=True)
        print(f"✅ Done: {data_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {data_path}\n{e}")

def list_stems(folder, exts=None):
    """
    返回指定目录下（不递归）的文件 stem 集合。
    exts 为 None 表示不过滤扩展名；否则只统计给定扩展名集合的文件。
    """
    stems = set()
    if not os.path.isdir(folder):
        return stems
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        name, ext = os.path.splitext(f)
        if exts is None or ext.lower() in exts:
            stems.add(name)
    return stems

def has_complete_depth(data_path, disp_path):
    """
    判断 disp_path 是否已经对 data_path 下的每张输入图像都生成了对应的深度图。
    规则：只要 disp 下存在与输入图像同名（stem 相同）的任意文件，即视为已生成。
    """
    img_stems = list_stems(data_path, IMG_EXTS)
    if not img_stems:
        return False  # 没有输入图像则不算完成

    disp_stems = list_stems(disp_path, exts=None)  # 不限制扩展名
    missing = img_stems - disp_stems
    if missing:
        # 可选：打印缺失数量，避免刷屏不逐一列出文件名
        print(f"ℹ️  Incomplete: {disp_path} missing {len(missing)} files.")
        return False
    return True

def collect_tasks(root, encoder, input_size, pred_only, grayscale, skip_if_exists=True):
    """遍历 KITTI 目录结构并生成命令任务"""
    tasks = []
    for date_dir in sorted(os.listdir(root)):
        date_path = os.path.join(root, date_dir)
        if not os.path.isdir(date_path):
            continue

        for drive_dir in sorted(os.listdir(date_path)):
            if not drive_dir.endswith("_sync"):
                continue

            drive_path = os.path.join(date_path, drive_dir)

            for cam in ["image_02", "image_03"]:
                data_path = os.path.join(drive_path, cam, "data")
                if not os.path.isdir(data_path):
                    continue

                disp_path = os.path.join(drive_path, cam, "disp")
                os.makedirs(disp_path, exist_ok=True)

                # 如果已完整生成，则跳过
                if skip_if_exists and has_complete_depth(data_path, disp_path):
                    print(f"⏭️  Skip (already complete): {data_path} -> {disp_path}")
                    continue

                cmd = [
                    "python", "run.py",
                    "--encoder", encoder,
                    "--img-path", data_path,
                    "--outdir", disp_path,
                    "--input-size", str(input_size)
                ]
                if pred_only:
                    cmd.append("--pred-only")
                if grayscale:
                    cmd.append("--grayscale")

                tasks.append((cmd, data_path))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel Batch Runner for Depth Anything on KITTI")
    parser.add_argument("--root", type=str, required=True, help="Path to KITTI raw root folder")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=1216)
    parser.add_argument("--pred-only", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    # 新增开关：默认跳过已完成的文件夹；想强制重跑时加 --no-skip-if-exists
    parser.add_argument("--no-skip-if-exists", action="store_true",
                        help="Do not skip folders whose disp already contains all corresponding depth maps")

    args = parser.parse_args()
    skip_if_exists = not args.no_skip_if_exists

    # 收集任务
    tasks = collect_tasks(
        args.root, args.encoder, args.input_size, args.pred_only, args.grayscale,
        skip_if_exists=skip_if_exists
    )

    if not tasks:
        print("⚠️ No tasks to run (either no valid image folders found, or all already complete).")
        return

    print(f"🔍 Found {len(tasks)} tasks, starting with {args.workers} parallel workers...")

    # 并行执行
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_task, cmd, path) for cmd, path in tasks]

        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except Exception as e:
                print(f"⚠️ Task {i} failed with exception: {e}")

    print("\n🎉 All tasks completed!")


if __name__ == "__main__":
    main()
