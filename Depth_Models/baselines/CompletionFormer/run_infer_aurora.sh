#!/bin/bash
# Phase 2 inference: run the AURORA-trained CompletionFormer on TEST + DENSE.
# Usage: bash run_infer_aurora.sh [checkpoint.pt]
#   default = latest available experiments_aurora/model_*.pt (16-epoch fast schedule)
set -e
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
export LD_LIBRARY_PATH=/path/to/aurora-root/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH
DEFAULT_CK=$(ls -t /path/to/aurora-root/Code/CompletionFormer/experiments_aurora/model_*.pt 2>/dev/null | head -1)
CK=${1:-$DEFAULT_CK}
PRE=/path/to/aurora-root/PhD_Project_2025/Experiments_Plan/results_zeroshot_benchmark/preds
SC=/path/to/aurora-root/Code/CompletionFormer/infer_.py
DAT=/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti
cd /path/to/aurora-root/Code/CompletionFormer/src   # cwd=src for ./pretrained/pvt.pth
echo "Using checkpoint: $CK"
python $SC --split TEST  --root $DAT/val_selection_cropped_TEST  --out $PRE/CompletionFormer-AURORA__TEST  --pretrain $CK
python $SC --split DENSE --root $DAT/val_selection_cropped_DENSE --out $PRE/CompletionFormer-AURORA__DENSE --pretrain $CK
echo "AURORA inference DONE"
