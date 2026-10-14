import os
import glob
import argparse

import numpy as np
import torch
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV
from unik3d.utils.visualization import colorize


def save_depth_metric_x256_u16(depth_torch: torch.Tensor, out_path: str):
    """
    保存为 16-bit 单通道 PNG:
    像素值 = metric depth(米) * 256
    读取时 depth(米) = 像素值 / 256
    """
    d = depth_torch.squeeze().detach().cpu().numpy().astype(np.float32)

    # 清理 NaN/inf
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # 防止 uint16 溢出：最大约 65535/256 ≈ 255.996m
    d = np.clip(d, 0.0, 65535.0 / 256.0)

    d_u16 = np.round(d * 256.0).astype(np.uint16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(d_u16, mode="I;16").save(out_path)


def read_kitti_intrinsics_txt(intri_path: str):
    """
    读取你给的 9 个数格式：
    fx 0 cx  0 fy cy  0 0 1
    """
    with open(intri_path, "r") as f:
        vals = [float(x) for x in f.read().strip().split()]
    if len(vals) != 9:
        raise ValueError(f"Expected 9 floats in intrinsics txt, got {len(vals)}: {intri_path}")
    K = np.array(vals, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy


@torch.no_grad()
def infer_one(model: UniK3D, rgb_path: str, intri_path: str, device: torch.device):
    rgb = np.array(Image.open(rgb_path))
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    fx, fy, cx, cy = read_kitti_intrinsics_txt(intri_path)

    # 构造 OPENCV params，后面的畸变项 KITTI 这份文件没有 => 全补 0
    # 与 demo 的 dl3dv.json 参数长度一致：16
    params = torch.tensor([fx, fy, cx, cy] + [0.0] * 12, dtype=torch.float32)
    camera = OPENCV(params=params)

    outputs = model.infer(rgb=rgb_torch.to(device), camera=camera, normalize=True)
    return rgb_torch, outputs


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
    parser.add_argument(
        "--out_dir",
        default="/path/to/aurora-root/Code/UniK3D/results",
    )
    parser.add_argument("--type", default="l", choices=["s", "b", "l"])
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--interpolation_mode", default="bilinear")
    parser.add_argument("--exts", default="png,jpg,jpeg,bmp")
    args = parser.parse_args()

    depth16_dir = os.path.join(args.out_dir, "depth16")
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(depth16_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("Torch version:", torch.__version__)
    name = f"unik3d-vit{args.type}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print("Device:", device)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.image_dir, f"*.{e}"))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {args.image_dir} with extensions: {exts}")

    missing = 0
    processed = 0

    for rgb_path in img_paths:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        intri_path = os.path.join(args.intrinsics_dir, stem + ".txt")

        if not os.path.exists(intri_path):
            print(f"[WARN] Missing intrinsics: {intri_path}")
            missing += 1
            continue

        rgb_torch, outputs = infer_one(model, rgb_path, intri_path, device)

        depth = outputs["depth"]  # metric depth (m)
        depth16_path = os.path.join(depth16_dir, stem+ ".png")
        save_depth_metric_x256_u16(depth, depth16_path)

        # 维持“原本 demo 的逻辑”做可视化（unik3d.utils.visualization.colorize）
        depth_np = depth.detach().cpu().numpy()
        vis = Image.fromarray(colorize(depth_np.squeeze()))
        vis_path = os.path.join(vis_dir, stem + "_depth_vis.png")
        vis.save(vis_path)

        processed += 1
        if processed % 50 == 0:
            print(f"[INFO] processed {processed}/{len(img_paths)}")

    print("Done.")
    print("Total images:", len(img_paths))
    print("Processed:", processed)
    print("Missing intrinsics:", missing)
    print("Saved depth16 to:", depth16_dir)
    print("Saved vis to:", vis_dir)


if __name__ == "__main__":
    main()
