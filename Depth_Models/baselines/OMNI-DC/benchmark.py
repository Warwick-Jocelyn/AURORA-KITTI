#!/usr/bin/env python3
"""
Benchmark OMNI-DC:
  - Parameters (M)
  - Processing Time (ms) at KITTI resolution (352x1216)
"""

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

sys.path.insert(0, "/path/to/aurora-root/Code/OMNI-DC/src")

import torch
import torch.nn as nn
import numpy as np

# Load model via the same path as demo.py
from config import args as args_config
from model.ognidc import OGNIDC

# ---- config ---------------------------------------------------------------
H, W = 352, 1216          # KITTI resolution
WARMUP_RUNS  = 10
MEASURE_RUNS = 100

# Patch args to match batch_demo settings
args_config.num_resolution = 3
args_config.load_dav2 = 1
args_config.gpus = "0"
args_config.GRU_iters = 1
args_config.optim_layer_input_clamp = 1.0
args_config.depth_activation_format = "exp"
args_config.whiten_sparse_depths = 1
args_config.gru_internal_whiten_method = "median"
args_config.backbone_mode = "rgbd"
args_config.pred_confidence_input = 1
args_config.multi_resolution_learnable_gradients_weights = "uniform"
args_config.max_depth = 300.0
args_config.data_normalize_median = 1

print("Loading model from HuggingFace (zuoym15/OMNI-DC)...")
net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=args_config)
net = nn.DataParallel(net)
net.eval()
net.cuda()

# ---- Parameter count -------------------------------------------------------
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\n{'='*45}")
print(f"Parameters (total)     : {total_params/1e6:.2f} M")
print(f"Parameters (trainable) : {trainable_params/1e6:.2f} M")

# ---- Dummy input -----------------------------------------------------------
# Pad H/W to be divisible by 4 * 2^(num_resolution-1) = 32
diviser = int(4 * 2 ** (args_config.num_resolution - 1))
H_pad = (diviser - H % diviser) % diviser
W_pad = (diviser - W % diviser) % diviser
H_in, W_in = H + H_pad, W + W_pad

rgb   = torch.randn(1, 3, H_in, W_in).cuda()
dep   = torch.randn(1, 1, H_in, W_in).cuda().clamp(min=0)
K     = torch.eye(3).reshape(1, 3, 3).cuda()

sample = {'rgb': rgb, 'dep': dep, 'K': K, 'pattern': 0}

# ---- Warm-up ---------------------------------------------------------------
print(f"\nInput size : {H}x{W} (padded to {H_in}x{W_in})")
print(f"Warm-up runs : {WARMUP_RUNS}")
print(f"Measure runs : {MEASURE_RUNS}")
print("Warming up...")

with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        _ = net(sample)
torch.cuda.synchronize()

# ---- Timing ----------------------------------------------------------------
print("Measuring...")
times = []
with torch.no_grad():
    for _ in range(MEASURE_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = net(sample)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

times = np.array(times)
print(f"\n{'='*45}")
print(f"Processing Time @ {H}x{W}")
print(f"  Mean   : {times.mean():.1f} ms")
print(f"  Median : {np.median(times):.1f} ms")
print(f"  Std    : {times.std():.1f} ms")
print(f"  Min    : {times.min():.1f} ms")
print(f"  Max    : {times.max():.1f} ms")
print(f"  FPS    : {1000/times.mean():.1f}")
print(f"{'='*45}")

# GPU info
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"CUDA memory reserved : {torch.cuda.memory_reserved()/1e9:.2f} GB")
