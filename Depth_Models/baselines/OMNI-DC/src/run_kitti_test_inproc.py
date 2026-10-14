"""
In-process OMNI-DC runner for val_selection_cropped_TEST.

Why this exists: ../run_kitti_test.py spawns `python demo.py` per frame,
which costs ~15 s of import + ckpt load *per image*. Doing 1600 frames that
way is 6-9 hours. Loading the model once and looping is ~10-25 min.

Also fixes the saturation problem: original demo.py saves
`(depth_pred * 1000).clip(0, 65535)` -> hard cap at 65.535 m. KITTI GT goes
to ~84 m. We rescale to *256 (cap at 65535/256 ≈ 255.996 m), matching the
DA3 / UniK3D convention so all 4 PNG-based methods share one decode rule.

Output: <output_dir>/<stem>.png  -- 16-bit metric depth (1 px = depth_m * 256)
        <output_dir>/_run_log.csv -- per-frame ms timing
"""
import os
import sys
import time
import argparse
import csv
from pathlib import Path

# ---- our own arg parsing FIRST, then construct the argv that OMNI-DC
# config.py will parse on import (it parses sys.argv when imported). ----
my_parser = argparse.ArgumentParser()
my_parser.add_argument("--rgb_dir", required=True)
my_parser.add_argument("--depth_dir", required=True)
my_parser.add_argument("--depth_ext", default=".png",
                       help="extension of sparse-depth files in --depth_dir (.png or .npy). "
                            "PNG path matches demo.py decoding (16-bit /256). "
                            "Use .npy only when depth_dir contains float32 metres .npy files.")
my_parser.add_argument("--out_dir", required=True)
my_parser.add_argument("--ckpt", default="../checkpoints/modelv1.1_best_72epochs.pt")
my_parser.add_argument("--png_scale", type=float, default=256.0,
                       help="meters -> uint16 multiplier. 256 ≈ 256m max range.")
my_parser.add_argument("--limit", type=int, default=0, help="<=0 means all")
my_parser.add_argument("--start", type=int, default=0, help="resume index")
my_parser.add_argument("--overwrite", action="store_true")
my_args, _ = my_parser.parse_known_args()

# Forge the argv that OMNI-DC config will parse. Mirrors run_kitti_test.py DEMO_ARGS.
sys.argv = [
    "run_kitti_test_inproc.py",
    "--max_depth", "300.0",
    "--data_normalize_median", "1",
    "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform",
    "--load_dav2", "1",
    "--gpus", "0",
    "--GRU_iters", "1",
    "--optim_layer_input_clamp", "1.0",
    "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1",
    "--gru_internal_whiten_method", "median",
    "--backbone_mode", "rgbd",
    "--pred_confidence_input", "1",
    "--pretrain", my_args.ckpt,
    # demo expects --input_rgb / --input_depth even though we override per frame
    "--input_rgb", "/dev/null",
    "--input_depth", "/dev/null",
    "--output_dir", my_args.out_dir,
    "--output_name", "dummy",
]

# Now import config (parses sys.argv) and rest of the deps.
from config import args as args_config
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from model.ognidc import OGNIDC

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model(args):
    if args.pretrain is not None:
        net = OGNIDC(args)
        ckpt = torch.load(args.pretrain, map_location="cpu")
        key_m, key_u = net.load_state_dict(ckpt["net"], strict=False)
        if key_m:
            print(f"[warn] missing keys: {len(key_m)}")
        print(f"[ok] checkpoint loaded from {args.pretrain}")
    else:
        net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=args)
    net = nn.DataParallel(net)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[ok] params: {n_params/1e6:.2f} M")
    return net, n_params


t_rgb = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
t_dep = T.Compose([T.ToTensor()])


def infer_one(net, args, rgb_path, depth_path):
    rgb = t_rgb(Image.open(rgb_path))
    if str(depth_path).lower().endswith(".npy"):
        depth_arr = np.load(depth_path)
    else:
        depth_arr = np.array(Image.open(depth_path)).astype(np.float32) / 256.0
    dep_sp = t_dep(depth_arr)

    rgb = rgb[None].cuda()
    dep_sp = dep_sp[None].cuda()
    K = torch.eye(3).reshape(1, 3, 3).cuda()
    sample = {"rgb": rgb, "dep": dep_sp, "K": K, "pattern": 0}

    _, _, H, W = rgb.shape
    diviser = int(4 * 2 ** (args.num_resolution - 1))
    H_pad = (diviser - H % diviser) % diviser
    W_pad = (diviser - W % diviser) % diviser
    if H_pad or W_pad:
        sample["rgb"] = torch.nn.functional.pad(rgb, (0, W_pad, 0, H_pad))
        sample["dep"] = torch.nn.functional.pad(dep_sp, (0, W_pad, 0, H_pad))

    with torch.no_grad():
        out = net(sample)
    pred = out["pred"][..., :H, :W].squeeze().cpu().numpy()
    return pred  # metres, float


def main():
    out_dir = Path(my_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgbs = sorted(Path(my_args.rgb_dir).glob("*.png"))
    if my_args.limit > 0:
        rgbs = rgbs[: my_args.limit]
    rgbs = rgbs[my_args.start:]
    print(f"[start] {len(rgbs)} frames -> {out_dir}")
    print(f"[note]  PNG scale = pixel = depth_m * {my_args.png_scale}  (max {65535/my_args.png_scale:.1f} m)")

    net, n_params = build_model(args_config)

    # warmup
    if rgbs:
        rgb_path = str(rgbs[0])
        dep_path = Path(my_args.depth_dir) / (rgbs[0].stem.replace("_sync_image_", "_sync_velodyne_raw_") + my_args.depth_ext)
        if dep_path.exists():
            t0 = time.time()
            _ = infer_one(net, args_config, rgb_path, str(dep_path))
            torch.cuda.synchronize()
            print(f"[warmup] {(time.time()-t0)*1000:.0f} ms")

    log_path = out_dir / "_run_log.csv"
    with open(log_path, "w", newline="") as logf:
        wlog = csv.writer(logf)
        wlog.writerow(["idx", "stem", "ms_total", "ms_gpu", "ok"])
        ok = 0
        skipped = 0
        t_overall = time.time()
        for i, rgb_path in enumerate(tqdm(rgbs, dynamic_ncols=True)):
            stem = rgb_path.stem
            out_png = out_dir / f"{stem}.png"
            if out_png.exists() and not my_args.overwrite:
                skipped += 1
                continue
            dep_name = stem.replace("_sync_image_", "_sync_velodyne_raw_") + my_args.depth_ext
            dep_path = Path(my_args.depth_dir) / dep_name
            if not dep_path.exists():
                wlog.writerow([i, stem, -1, -1, "missing_depth"])
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                pred = infer_one(net, args_config, str(rgb_path), str(dep_path))
            except Exception as e:
                wlog.writerow([i, stem, -1, -1, f"err:{type(e).__name__}"])
                continue
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            u16 = np.clip(pred * my_args.png_scale, 0, 65535).astype(np.uint16)
            Image.fromarray(u16).save(out_png)
            wlog.writerow([i, stem, f"{ms:.1f}", f"{ms:.1f}", "ok"])
            ok += 1
            if (i + 1) % 100 == 0:
                logf.flush()
        dt = time.time() - t_overall
    print(f"[done] ok={ok}, skipped={skipped}, total {dt/60:.1f} min")
    print(f"[done] log: {log_path}")


if __name__ == "__main__":
    main()
