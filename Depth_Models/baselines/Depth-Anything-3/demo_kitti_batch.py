import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 允许的输入图像扩展名
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def run_single_task(cmd, data_path):
    """执行单个命令"""
    try:
        print(f"🚀 Start: {data_path}")
        subprocess.run(cmd, check=True)
        print(f"✅ Done:  {data_path}")
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


def has_complete_depth(data_path, disp3_path):
    """
    判断 disp3_path 是否已经对 data_path 下的每张输入图像都生成了对应的深度图。
    规则：只要 DA3 下存在与输入图像同名（stem 相同）的任意文件，即视为已生成。
    """
    img_stems = list_stems(data_path, IMG_EXTS)
    if not img_stems:
        return False  # 没有输入图像则不算完成

    disp_stems = list_stems(disp3_path, exts=None)  # 不限制扩展名
    missing = img_stems - disp_stems
    if missing:
        print(f"ℹ️  Incomplete: {disp3_path} missing {len(missing)} files.")
        return False
    return True


def collect_tasks(
    root,
    out_folder_name,
    workers_batch_size,
    ckpt,
    process_res_method,
    process_res,
    auto_process_res,
    cap,
    align_back_to_input,
    u16_mode,
    invert,
    clamp_min,
    clamp_max,
    png_scale,
    save_npz,
    skip_if_exists=True,
):
    """
    遍历 KITTI raw 目录结构并生成任务。
    每个任务：对一个 data_path -> disp3_path 跑一次 da3_batch_depth_u16.py
    """
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

                disp3_path = os.path.join(drive_path, cam, out_folder_name)
                os.makedirs(disp3_path, exist_ok=True)

                if skip_if_exists and has_complete_depth(data_path, disp3_path):
                    print(f"⏭️  Skip (already complete): {data_path} -> {disp3_path}")
                    continue

                cmd = [
                    "python", "da3_batch_depth_u16.py",
                    "--image-dir", data_path,
                    "--out-dir", disp3_path,
                    "--batch-size", str(workers_batch_size),
                    "--ckpt", ckpt,
                    "--process-res-method", process_res_method,
                    "--u16-mode", u16_mode,
                    "--clamp-min", str(clamp_min),
                    "--clamp-max", str(clamp_max),
                    "--png-scale", str(png_scale),
                ]

                # resolution strategy
                if process_res > 0:
                    cmd += ["--process-res", str(process_res)]
                else:
                    cmd += ["--process-res", "0", "--auto-process-res", auto_process_res, "--cap", str(cap)]

                # flags
                if align_back_to_input:
                    cmd.append("--align-back-to-input")
                if invert:
                    cmd.append("--invert")
                if save_npz:
                    cmd.append("--save-npz")

                tasks.append((cmd, data_path))

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel Batch Runner for Depth Anything 3 on KITTI Raw -> DA3")

    parser.add_argument("--root", type=str, required=True, help="Path to KITTI raw root folder")

    # output folder name
    parser.add_argument("--out-folder-name", type=str, default="DA3",
                        help='Name of output folder under each camera path (default: "DA3").')

    # parallel
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (folders in parallel)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size inside each folder inference")

    # skip behavior
    parser.add_argument("--no-skip-if-exists", action="store_true",
                        help="Do not skip folders whose DA3 already contains all corresponding depth maps")

    # model/inference options (forwarded to da3_batch_depth_u16.py)
    parser.add_argument("--ckpt", type=str, default="depth-anything/da3nested-giant-large")
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--process-res", type=int, default=0,
                        help="Fixed process_res for all images. If 0, use auto strategy per-image (capped).")
    parser.add_argument("--auto-process-res", type=str, default="max_side_cap", choices=["max_side", "max_side_cap"])
    parser.add_argument("--cap", type=int, default=4096)

    parser.add_argument("--align-back-to-input", action="store_true",
                        help="If output depth size != input size, interpolate back to (H,W).")

    # output encoding
    parser.add_argument("--u16-mode", type=str, default="minmax", choices=["minmax", "clamp_scale"])
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=255.0)
    parser.add_argument("--png-scale", type=float, default=256.0)
    parser.add_argument("--save-npz", action="store_true")

    args = parser.parse_args()
    skip_if_exists = not args.no_skip_if_exists

    tasks = collect_tasks(
        root=args.root,
        out_folder_name=args.out_folder_name,
        workers_batch_size=args.batch_size,
        ckpt=args.ckpt,
        process_res_method=args.process_res_method,
        process_res=args.process_res,
        auto_process_res=args.auto_process_res,
        cap=args.cap,
        align_back_to_input=args.align_back_to_input,
        u16_mode=args.u16_mode,
        invert=args.invert,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        png_scale=args.png_scale,
        save_npz=args.save_npz,
        skip_if_exists=skip_if_exists,
    )

    if not tasks:
        print("⚠️ No tasks to run (either no valid image folders found, or all already complete).")
        return

    print(f"🔍 Found {len(tasks)} tasks, starting with {args.workers} parallel workers...")

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
