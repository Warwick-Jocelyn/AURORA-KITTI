#!/usr/bin/env python3
"""
infer_on_dmd3c_val.py
=====================
Run a trained Sparse-to-Dense (Fangchang Ma) checkpoint on the DMD3C
weather-corrupted VAL split and save 16-bit depth PNGs for downstream
evaluation (e.g. eval_denoise.py).

Usage
-----
  # rgbd smoke (50 frames):
  python infer_on_dmd3c_val.py --input rgbd --limit 50 \
      --out_dir /tmp/s2d_preds_rgbd_smoke

  # full val, depth-only:
  python infer_on_dmd3c_val.py --input d \
      --out_dir /tmp/s2d_preds_d

NOTES on normalization (verified against kitti_loader.py + transforms.py):
  * val='select' uses no_transform  → images arrive at their native 352x1216
  * rgb_read returns uint8 HxWx3 → to_float_tensor casts to float32 in [0,255]
    The model was trained with RGB in [0,255] (no /255 normalisation anywhere in
    the original pipeline; NormalizeNumpyArray is defined but never called in val).
  * depth_read: depth_png.astype(int) / 256.0  → float metres, shape HxWx1
  * to_float_tensor transposes HxWxC → CxHxW

Input zeroing for modality ablations:
  --input rgb  → d tensor zeroed
  --input d    → rgb (and g) tensor zeroed
  --input rgbd → nothing zeroed (full model)

Output
------
  <out_dir>/<drive>_groundtruth_depth_<frame>_image_<cam>.png
  encoded as uint16: round(metres * 256), matching the GT encoding.
"""

import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ── Must run from repo root (dataloaders/calib_cam_to_cam.txt is hardcoded) ──
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from model import DepthCompletionNet  # noqa: E402

VAL_ROOT = (
    "/path/to/aurora-root/PhD_Project_2025"
    "/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_VAL"
)
CKPT_DEFAULT = (
    "/path/to/aurora-root/PhD_Project_2025"
    "/self-supervised-depth-completion/checkpoints_AURORA/supervised/model_best.pth.tar"
)

# Target spatial size the model expects (and that DMD3C val already provides).
OHEIGHT, OWIDTH = 352, 1216


# ─────────────────────────────────────────────────────────────────────────────
# Minimal arg-namespace that satisfies DepthCompletionNet.__init__
# ─────────────────────────────────────────────────────────────────────────────
class _ModelArgs:
    """Tiny stand-in for argparse.Namespace; only fields model.py cares about."""
    def __init__(self, layers: int, input_mode: str, pretrained: bool = False):
        self.layers = layers
        self.input = input_mode
        self.pretrained = pretrained


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers — replicate kitti_loader exactly
# ─────────────────────────────────────────────────────────────────────────────
def rgb_read(path: str) -> np.ndarray:
    """Load RGB as uint8 HxWx3, same as kitti_loader.rgb_read."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype="uint8")
    img.close()
    return arr


def depth_read(path: str) -> np.ndarray:
    """Load 16-bit depth PNG → float32 metres HxWx1, same as kitti_loader.depth_read."""
    img = Image.open(path)
    arr = np.array(img, dtype=int)
    img.close()
    # The original code asserts max>255; DMD3C GTs are 16-bit so this holds.
    depth = arr.astype(np.float32) / 256.0
    return np.expand_dims(depth, -1)  # HxWx1


def to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """HxWxC numpy → CxHxW float32 tensor (matches kitti_loader.to_float_tensor)."""
    if arr.ndim == 3:
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
    elif arr.ndim == 2:
        return torch.from_numpy(arr.copy()).float()
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim}")


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────────────────────────────────────
def gt_basename_from_velodyne(vel_name: str) -> str:
    """
    vel_name  : <drive>_velodyne_raw_<frame>_image_<cam>.png
    returns   : <drive>_groundtruth_depth_<frame>_image_<cam>.png
    """
    return vel_name.replace("_velodyne_raw_", "_groundtruth_depth_")


def image_basename_from_velodyne(vel_name: str) -> str:
    """
    vel_name  : <drive>_velodyne_raw_<frame>_image_<cam>.png
    returns   : <drive>_image_<frame>_image_<cam>.png
    """
    return vel_name.replace("_velodyne_raw_", "_image_")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Sparse-to-Dense inference on DMD3C VAL")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT,
                    help="path to model_best.pth.tar")
    ap.add_argument("--input", default="rgbd", choices=["rgbd", "rgb", "d"],
                    help="modality: zero-out unused channels for ablation")
    ap.add_argument("--layers", type=int, default=34,
                    help="ResNet depth (must match checkpoint; default 34)")
    ap.add_argument("--out_dir", required=True,
                    help="directory to write predicted 16-bit PNGs")
    ap.add_argument("--val_root", default=VAL_ROOT,
                    help="path to val_selection_cropped_VAL/")
    ap.add_argument("--limit", type=int, default=0,
                    help="max frames to process (0 = all 1600)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[infer] device={device}  input_mode={args.input}  ckpt={args.ckpt}")

    # ── Build model ─────────────────────────────────────────────────────────
    # Build with the SAME modality the checkpoint was trained with. Training
    # builds a TRUE per-modality net (rgb-only has no conv1_d; d-only has no
    # conv1_img), so the state_dict only loads if input_mode == args.input.
    model_args = _ModelArgs(layers=args.layers, input_mode=args.input, pretrained=False)
    model = DepthCompletionNet(model_args)

    print(f"[infer] loading checkpoint …")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"[infer] checkpoint epoch={ckpt.get('epoch', '?')} loaded OK")

    model = model.to(device)
    model.eval()

    # ── Modality flags ───────────────────────────────────────────────────────
    # The per-modality model reads ONLY the keys its branches need (rgbd: rgb+d;
    # rgb: rgb; d: d). We provide all keys in the batch; unused ones are ignored
    # by the model exactly as in training. No tensor-zeroing needed (and zeroing
    # would NOT match training, which drops the modality at the architecture level).
    use_rgb = ("rgb" in args.input)
    use_d   = ("d"   in args.input)

    # ── Discover VAL frames via velodyne_raw ─────────────────────────────────
    vel_dir = os.path.join(args.val_root, "velodyne_raw")
    img_dir = os.path.join(args.val_root, "image")
    gt_dir  = os.path.join(args.val_root, "groundtruth_depth")

    vel_files = sorted(os.listdir(vel_dir))
    if args.limit > 0:
        vel_files = vel_files[: args.limit]
    print(f"[infer] frames to process: {len(vel_files)}")

    n_ok = n_skip = 0

    with torch.no_grad():
        for idx, vel_fn in enumerate(vel_files):
            if not vel_fn.endswith(".png"):
                n_skip += 1
                continue

            vel_path = os.path.join(vel_dir, vel_fn)
            img_path = os.path.join(img_dir, image_basename_from_velodyne(vel_fn))
            out_name = gt_basename_from_velodyne(vel_fn)
            out_path = os.path.join(args.out_dir, out_name)

            # ── Load inputs ──────────────────────────────────────────────
            try:
                # Sparse depth  (HxWx1 float metres)
                sparse_np = depth_read(vel_path)
                # RGB  (HxWx3 uint8)
                rgb_np = rgb_read(img_path) if os.path.exists(img_path) else None
            except Exception as e:
                print(f"[WARN] skip {vel_fn}: {e}")
                n_skip += 1
                continue

            if rgb_np is None and use_rgb:
                print(f"[WARN] skip {vel_fn}: no RGB found at {img_path}")
                n_skip += 1
                continue

            # ── Verify spatial size ───────────────────────────────────────
            h, w = sparse_np.shape[:2]
            if (h, w) != (OHEIGHT, OWIDTH):
                # The repo uses no_transform for val='select'; DMD3C val frames
                # are pre-cropped to 352x1216, so this should never trigger.
                print(f"[WARN] unexpected size {h}x{w} for {vel_fn}, skipping")
                n_skip += 1
                continue

            # ── to_float_tensor (replicates kitti_loader exactly) ─────────
            d_t = to_float_tensor(sparse_np)   # 1xHxW, float metres
            if rgb_np is not None:
                rgb_t = to_float_tensor(rgb_np)  # 3xHxW, float in [0,255]
            else:
                rgb_t = torch.zeros(3, OHEIGHT, OWIDTH)

            # Gray = mean of RGB channels (replicate handle_gray with use_g=True)
            gray_np = np.expand_dims(
                np.array(Image.fromarray(rgb_np).convert("L"), dtype=np.uint8) if rgb_np is not None
                else np.zeros((OHEIGHT, OWIDTH), dtype=np.uint8),
                -1
            )
            g_t = to_float_tensor(gray_np)  # 1xHxW

            # No tensor-zeroing: the per-modality model reads only the keys its
            # branches need (matches training, which drops the modality at the
            # architecture level).

            # ── Build batch dict (unsqueeze to add batch dim) ─────────────
            batch = {
                "rgb": rgb_t.unsqueeze(0).to(device),  # 1x3xHxW
                "d":   d_t.unsqueeze(0).to(device),    # 1x1xHxW
                "g":   g_t.unsqueeze(0).to(device),    # 1x1xHxW
            }

            # ── Forward pass ─────────────────────────────────────────────
            pred = model(batch)  # 1x1xHxW, metres (clamped to >=0.9m in eval)
            pred_np = pred.squeeze().cpu().numpy()  # HxW

            # ── Save as 16-bit PNG: round(metres * 256) ──────────────────
            pred_u16 = np.clip(np.round(pred_np * 256.0), 0, 65535).astype(np.uint16)
            Image.fromarray(pred_u16).save(out_path)

            n_ok += 1
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(vel_files)}] {out_name}  "
                      f"pred range [{pred_np.min():.2f}, {pred_np.max():.2f}] m")

    print(f"\n[infer] done — saved={n_ok}  skipped={n_skip}  out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
