#!/bin/bash
# PROPER full-schedule retrain of CompletionFormer on AURORA ().
# Difference vs the fast run: FULL 40,192-frame set every epoch (not a 10k subset),
# 12 epochs, lr 1e-5 step-decay x0.1 every 5 epochs (-> 1e-5 ep0-4, 1e-6 ep5-9,
# 1e-7 ep10-11). Same recipe otherwise: from-scratch, RGB+D, full 352x1216,
# CF-native L1+L2 + SSI(0.1)+grad(0.05) clean-da3 distillation, Adam, batch1+accum4.
# ~2.8 h/epoch at ~4.1 it/s, peak ~9 GB -> ~33 h total. Robust to VS Code close (tmux).
set -e
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
export LD_LIBRARY_PATH=/path/to/aurora-root/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUT=/path/to/aurora-root/Code/CompletionFormer/experiments_aurora_full
mkdir -p "$OUT"
cd /path/to/aurora-root/Code/CompletionFormer/src   # cwd=src so backbone finds ./pretrained/pvt.pth
python -u /path/to/aurora-root/Code/CompletionFormer/train_aurora.py \
    --epochs 12 --batch 1 --accum 4 --subset 0 --lr 1e-5 --workers 8 \
    --ssi 0.1 --grad 0.05 --mem_frac 0.72 --seed 43 \
    --out "$OUT"
echo "CF FULL TRAINING COMPLETE -> $OUT"
