#!/bin/bash
# ============================================================================
# Normalization-isolation ablation for RC "Performance gains not fully explained".
# Identical recipe to the main DDCD (DMD3D_BP_KITTI_v4): data=KITTI (h256), bs=2,
# CLEAN da3 teacher (kitti.py Try1), 30 epochs, lambda_distill=0.1/lambda_grad=0.05.
# ONLY difference: ++no_teacher_norm=true  -> skips the GT-aligned teacher
# normalisation (Eq.6, disp_to_depth_proxy); raw teacher goes to the SSI loss.
# The gap between this run and full DDCD (1801.93 mm) = the normalisation contribution.
#
# Uses train_ablation_nonorm.py (a copy of train.py + the no_teacher_norm branch),
# so train.py is left untouched.
#
# PREREQUISITE: free the GPU first (the 4090 is held by tmux ddcd_h352). A second
# 30-epoch run will OOM alongside it. Recommended: stop h352 (dead-end/overfit),
# then launch this. ETA ~2-3 days (h256 bs=2, 30 ep).
# ============================================================================
set -e
cd /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=$(realpath ..):   # CRITICAL: matches h352's env so `import utils` resolves to
                                    # BP-Net/DMD3C/utils.py (handles a full .pth chpt), not BP-Net/utils.py
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python  # same env the main/h352 runs used (has wandb+torch+cuda)
NAME=DDCD_ablation_nonorm
mkdir -p runs/$NAME
nohup $PY -u train_ablation_nonorm.py \
  gpus=[0] num_workers=4 name=$NAME \
  ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
  net=PMP data=KITTI \
  lr=5e-4 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=30 test_epoch=1 ++net.sbn=false metric=MetricALL \
  ++lambda_distill=0.1 ++lambda_grad=0.05 \
  ++no_teacher_norm=true \
  ++max_train_images=full ++max_val_images=1600 \
  ++use_wandb=false \
  > runs/$NAME/train.log 2>&1 &
echo "launched $NAME (PID $!) -> runs/$NAME/train.log ; best -> checkpoint/$NAME/best_ema.pth"
