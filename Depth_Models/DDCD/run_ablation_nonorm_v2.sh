#!/bin/bash
# ============================================================================
# V2-teacher normalization-isolation ablation (headline-consistent).
# Identical recipe to run_ablation_nonorm.sh EXCEPT the clean teacher is
# Depth-Anything-V2 (data=KITTI_V2CLEAN -> kitti_v2clean.py loads the clean
# V2 disparity from kitti_depth_DC/.../disp/, David-confirmed provenance),
# matching the paper's headline DDCD (V2). ++no_teacher_norm=true skips the
# GT-aligned teacher normalisation (Eq.6); the gap vs full V2-DDCD
# (same-pipeline 1791.33 mm) = the normalisation contribution, cleanly
# isolated (same teacher version on both rows).
#
# Runs in FOREGROUND (no nohup/&) so a tmux session holds it -> survives
# VSCode/SSH exit. Watchdog (watchdog_nonorm_v2.py) early-stops on plateau.
# ============================================================================
set -e
cd /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=$(realpath ..):   # CRITICAL: `import utils` -> BP-Net/DMD3C/utils.py (handles full .pth chpt)
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
NAME=DDCD_ablation_nonorm_v2
mkdir -p runs/$NAME
echo "[launch] $NAME  data=KITTI_V2CLEAN (clean DA-V2 teacher)  no_teacher_norm=true  $(date)"
$PY -u train_ablation_nonorm.py \
  gpus=[0] num_workers=4 name=$NAME \
  ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
  net=PMP data=KITTI_V2CLEAN \
  lr=5e-4 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=30 test_epoch=1 ++net.sbn=false metric=MetricALL \
  ++lambda_distill=0.1 ++lambda_grad=0.05 \
  ++no_teacher_norm=true \
  ++max_train_images=full ++max_val_images=1600 \
  ++use_wandb=false \
  > runs/$NAME/train.log 2>&1
echo "[done] $NAME exited $(date)"
