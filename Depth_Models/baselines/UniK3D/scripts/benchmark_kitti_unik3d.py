import os
import glob
import time
import argparse

import numpy as np
import torch
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV


def read_kitti_intrinsics_txt(intri_path: str):
    with open(intri_path, "r") as f:
        vals = [float(x) for x in f.read().strip().split()]
    if len(vals) != 9:
        raise ValueError(f"Expected 9 floats, got {len(vals)}: {intri_path}")
    K = np.array(vals, dtype=np.float32).reshape(3, 3)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy


def build_opencv_camera_from_kitti(intri_path: str, device: torch.device):
    fx, fy, cx, cy = read_kitti_intrinsics_txt(intri_path)
    params = torch.tensor([fx, fy, cx, cy] + [0.0] * 12, dtype=torch.float32, device=device)
    return OPENCV(params=params)


def load_rgb_as_chw(rgb_path: str) -> torch.Tensor:
    rgb = np.array(Image.open(rgb_path))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def count_params_and_size(model: torch.nn.Module):
    total = 0
    trainable = 0
    bytes_est = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        bytes_est += n * p.element_size()
    # buffers (e.g. running_mean) 通常很小，但也可以加上
    for b in model.buffers():
        bytes_est += b.numel() * b.element_size()
    return total, trainable, bytes_est


@torch.no_grad()
def time_infer_only(model, rgb_chw_cpu, camera, device):
    # 只计 infer（不含 PIL/np 读取，不含 tensor 构造）
    rgb = rgb_chw_cpu.to(device, non_blocking=True)
    _ = model.infer(rgb=rgb, camera=camera, normalize=True)


@torch.no_grad()
def time_end_to_end(model, rgb_path, intri_path, device):
    # 端到端：包含 PIL 读取 + tensor 构造 + camera 构造 + infer
    rgb_chw = load_rgb_as_chw(rgb_path).to(device)
    camera = build_opencv_camera_from_kitti(intri_path, device)
    _ = model.infer(rgb=rgb_chw, camera=camera, normalize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        default="/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image",
    )
    parser.add_argument(
        "--intrinsics_dir",
        default="/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/intrinsics",
    )
    parser.add_argument("--type", default="l", choices=["s", "b", "l"])
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--interpolation_mode", default="bilinear")
    parser.add_argument("--num_imgs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--mode", default="infer_only", choices=["infer_only", "end_to_end"])
    parser.add_argument("--exts", default="png,jpg,jpeg,bmp")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    name = f"unik3d-vit{args.type}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode
    model = model.to(device).eval()

    total_params, trainable_params, bytes_est = count_params_and_size(model)
    print(f"Params total: {total_params/1e6:.2f} M")
    print(f"Params trainable: {trainable_params/1e6:.2f} M")
    print(f"Estimated weights+buffers size (current dtype): {bytes_est/1024/1024:.1f} MB")

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.image_dir, f"*.{e}"))
    img_paths = sorted(img_paths)

    # 收集有 intrinsics 的样本
    pairs = []
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        intri = os.path.join(args.intrinsics_dir, stem + ".txt")
        if os.path.exists(intri):
            pairs.append((p, intri))
        if len(pairs) >= args.num_imgs:
            break

    if len(pairs) == 0:
        raise RuntimeError("No valid (image, intrinsics) pairs found.")

    print(f"Benchmarking {len(pairs)} images, mode={args.mode}, warmup={args.warmup}")

    # 预加载到 CPU tensor（infer_only 模式更稳定，减少 IO 干扰）
    cpu_rgbs = []
    cameras = []
    if args.mode == "infer_only":
        for rgb_path, intri_path in pairs:
            cpu_rgbs.append(load_rgb_as_chw(rgb_path))
            cameras.append(build_opencv_camera_from_kitti(intri_path, device))

    # warmup
    for i in range(args.warmup):
        idx = i % len(pairs)
        if args.mode == "infer_only":
            time_infer_only(model, cpu_rgbs[idx], cameras[idx], device)
        else:
            time_end_to_end(model, pairs[idx][0], pairs[idx][1], device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # timing
    times_ms = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for i in range(len(pairs)):
            starter.record()
            if args.mode == "infer_only":
                time_infer_only(model, cpu_rgbs[i], cameras[i], device)
            else:
                time_end_to_end(model, pairs[i][0], pairs[i][1], device)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        # CPU fallback（不推荐用来跟论文对比）
        for i in range(len(pairs)):
            t0 = time.perf_counter()
            if args.mode == "infer_only":
                time_infer_only(model, cpu_rgbs[i], cameras[i], device)
            else:
                time_end_to_end(model, pairs[i][0], pairs[i][1], device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    avg = float(np.mean(times_ms))
    std = float(np.std(times_ms))
    print(f"Per-image time: avg={avg:.2f} ms, std={std:.2f} ms")
    print("All times (ms):", [round(x, 2) for x in times_ms])


if __name__ == "__main__":
    main()
