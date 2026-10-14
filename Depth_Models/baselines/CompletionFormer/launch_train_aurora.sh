#!/bin/bash
# Phase 2: CompletionFormer trained-from-scratch on AURORA (mirrors Sparse2Dense recipe)
set -e
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
export LD_LIBRARY_PATH=/path/to/aurora-root/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /path/to/aurora-root/Code/CompletionFormer/src   # cwd=src so backbone finds ./pretrained/pvt.pth
# FAST schedule ( timeframe): fresh random 10k-frame subset per epoch
# (resampled each epoch, seed=43+epoch -> varied data over training), 16 epochs.
# Everything else identical: from-scratch, full RGB+D, full 352x1216, CF-native
# L1+L2 + SSI(0.1)+grad(0.05) distill vs clean-da3 teacher, Adam lr 1e-5 step decay.
# batch 1 + accum 4 = effective batch 4 (CF full-res is memory-heavy; crop256+batch2
# reached ~18GB reserved > 16GB target, so kept batch1+accum4: ~14GB, safe on shared GPU).
python -u /path/to/aurora-root/Code/CompletionFormer/train_aurora.py \
    --epochs 16 --batch 1 --accum 4 --subset 10000 --lr 1e-5 --workers 8 \
    --ssi 0.1 --grad 0.05 --mem_frac 0.72 \
    --out /path/to/aurora-root/Code/CompletionFormer/experiments_aurora
