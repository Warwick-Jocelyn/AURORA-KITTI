#!/usr/bin/env python3
"""
Run OMNI-DC v1.0 (no Depth Anything V2) on KITTI val_selection_cropped (DENSE).

Input:
  RGB:   .../image/{name}.png
  Depth: .../velodyne_raw/{name}.png  (16-bit, pixel = depth_m * 256)

Output:
  output_KITTI_DENSE_v10/{name}.png   (16-bit, 1 pixel = 0.001 m)
"""

import subprocess
from pathlib import Path

# ---- paths ----------------------------------------------------------------
RGB_DIR    = Path("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image")
DEPTH_DIR  = Path("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/velodyne_raw")
OUTPUT_DIR = Path("/path/to/aurora-root/Code/OMNI-DC/output_KITTI_DENSE_v10")
OMNI_SRC   = Path("/path/to/aurora-root/Code/OMNI-DC/src")
CKPT       = "../checkpoints/modelv1.0_best.pt"

# ---- demo.py args (v1.0: load_dav2=0) ------------------------------------
DEMO_ARGS = [
    "--max_depth", "300.0",
    "--data_normalize_median", "1",
    "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform",
    "--load_dav2", "0",          # v1.0: no Depth Anything V2 (single-channel depth)
    "--gpus", "0",
    "--GRU_iters", "1",
    "--optim_layer_input_clamp", "1.0",
    "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1",
    "--gru_internal_whiten_method", "median",
    "--backbone_mode", "rgbd",
    "--pred_confidence_input", "1",
    "--pretrain", CKPT,
]

# --------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rgb_files = sorted(RGB_DIR.glob("*.png"))
total = len(rgb_files)
print(f"Found {total} RGB images. Using v1.0 checkpoint (no DA2).\n")

processed, skipped, existing = 0, 0, 0

for i, rgb_path in enumerate(rgb_files, 1):
    stem = rgb_path.stem
    depth_path = DEPTH_DIR / f"{stem}.png"

    if not depth_path.exists():
        print(f"[SKIP {i}/{total}] No depth for: {stem}")
        skipped += 1
        continue

    final_16bit = OUTPUT_DIR / f"{stem}.png"
    if final_16bit.exists():
        print(f"[EXIST {i}/{total}] {stem}")
        existing += 1
        continue

    print(f"[{i}/{total}] {stem}")

    cmd = (
        ["python", "demo.py"]
        + DEMO_ARGS
        + [
            "--input_rgb",   str(rgb_path),
            "--input_depth", str(depth_path),
            "--output_dir",  str(OUTPUT_DIR),
            "--output_name", stem,
        ]
    )

    result = subprocess.run(cmd, cwd=str(OMNI_SRC))
    if result.returncode != 0:
        print(f"  [ERROR] demo.py failed for {stem}")
        continue

    src_16bit = OUTPUT_DIR / f"{stem}_16bit.png"
    if src_16bit.exists():
        src_16bit.rename(final_16bit)
    else:
        print(f"  [WARN] _16bit.png not found for {stem}")

    for ext in [".png", ".npy"]:
        tmp = OUTPUT_DIR / f"{stem}{ext}"
        if tmp.exists() and tmp != final_16bit:
            tmp.unlink()

    processed += 1

print(f"\nDone. processed={processed}, skipped={skipped}, already_exist={existing}")
print(f"Results saved to: {OUTPUT_DIR}")
