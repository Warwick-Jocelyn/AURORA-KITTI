#!/bin/bash
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
SPLIT=$1
DATA=/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped_${SPLIT}
OUT=/path/to/aurora-root/PhD_Project_2025/Experiments_Plan/results_zeroshot_benchmark/preds/NLSPN-ZeroShot__${SPLIT}
CKPT=/path/to/aurora-root/Code/NLSPN/results/NLSPN_KITTI_DC.pt
cd /path/to/aurora-root/Code/NLSPN/src
python infer_zeroshot.py \
  --gt_dir   $DATA/groundtruth_depth \
  --image_dir $DATA/image \
  --velo_dir  $DATA/velodyne_raw \
  --out_dir   $OUT \
  --ckpt      $CKPT --workers 16
echo "RUN_EXIT=$?"
