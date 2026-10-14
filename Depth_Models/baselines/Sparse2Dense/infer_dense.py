#!/usr/bin/env python3
"""
infer_dense.py — run a trained Sparse-to-Dense modality checkpoint on the
real DENSE (SeeingThroughFog) split prepared in val_selection_cropped_DENSE.
DENSE filenames are identical across image/velodyne_raw/groundtruth_depth
(e.g. scene1_day_clear_0.png). Sparse & GT are encoded *256 (verified).
Output: <out_dir>/<name>.png as uint16 round(metres*256).
Cross-dataset generalization test: model trained on AURORA-KITTI.
"""
import argparse, os, sys, numpy as np
from PIL import Image
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from model import DepthCompletionNet

DENSE_ROOT = ("/path/to/aurora-root/PhD_Project_2025/DMD3C/"
              "BP-Net/DMD3C/datas/kitti/val_selection_cropped_DENSE")
OHEIGHT, OWIDTH = 352, 1216


class _ModelArgs:
    def __init__(self, layers, input_mode, pretrained=False):
        self.layers = layers; self.input = input_mode; self.pretrained = pretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", default="rgbd", choices=["rgbd", "rgb", "d"])
    ap.add_argument("--layers", type=int, default=34)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--root", default=DENSE_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer-dense] device={device} input={args.input} ckpt={args.ckpt}")

    model = DepthCompletionNet(_ModelArgs(args.layers, args.input, False))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"[infer-dense] ckpt epoch={ckpt.get('epoch','?')} loaded")

    vel_dir = os.path.join(args.root, "velodyne_raw")
    img_dir = os.path.join(args.root, "image")
    files = sorted(f for f in os.listdir(vel_dir) if f.endswith(".png"))
    if args.limit > 0: files = files[:args.limit]
    print(f"[infer-dense] frames: {len(files)}")

    n_ok = n_skip = 0
    with torch.no_grad():
        for idx, fn in enumerate(files):
            vp = os.path.join(vel_dir, fn); ip = os.path.join(img_dir, fn)
            if not os.path.exists(ip):
                n_skip += 1; continue
            sparse = np.array(Image.open(vp), dtype=int).astype(np.float32) / 256.0  # metres
            rgb = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
            if sparse.shape[:2] != (OHEIGHT, OWIDTH):
                print(f"[WARN] {fn} size {sparse.shape}; skip"); n_skip += 1; continue
            d_t = torch.from_numpy(sparse[None]).float()
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
            g = np.array(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
            g_t = torch.from_numpy(g[None]).float()
            batch = {"rgb": rgb_t.unsqueeze(0).to(device),
                     "d":   d_t.unsqueeze(0).to(device),
                     "g":   g_t.unsqueeze(0).to(device)}
            pred = model(batch).squeeze().cpu().numpy()
            u16 = np.clip(np.round(pred * 256.0), 0, 65535).astype(np.uint16)
            Image.fromarray(u16).save(os.path.join(args.out_dir, fn))
            n_ok += 1
            if (idx + 1) % 200 == 0:
                print(f"  [{idx+1}/{len(files)}] {fn} range [{pred.min():.2f},{pred.max():.2f}]m")
    print(f"[infer-dense] done saved={n_ok} skipped={n_skip} out={args.out_dir}")


if __name__ == "__main__":
    main()
