#!/bin/bash
# Post-training: run the properly-retrained CompletionFormer on TEST+DENSE,
# overwrite the benchmark pred dirs, drop the stale JSONs so the eval+website
# pipeline recomputes the CompletionFormer-AURORA rows, then refresh website/tex.
# Usage: bash cf_post.sh /path/to/model_XX.pt
set -e
CK="$1"
[ -z "$CK" ] && CK=$(ls -t /path/to/aurora-root/Code/CompletionFormer/experiments_aurora_full/model_*.pt | head -1)
echo "[cf_post] checkpoint: $CK"
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
export LD_LIBRARY_PATH=/path/to/aurora-root/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
RZB=/path/to/aurora-root/PhD_Project_2025/Experiments_Plan/results_zeroshot_benchmark
SC=/path/to/aurora-root/Code/CompletionFormer/infer_.py
DAT=/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti
cd /path/to/aurora-root/Code/CompletionFormer/src
python $SC --split TEST  --root $DAT/val_selection_cropped_TEST  --out $RZB/preds/CompletionFormer-AURORA__TEST  --pretrain "$CK"
python $SC --split DENSE --root $DAT/val_selection_cropped_DENSE --out $RZB/preds/CompletionFormer-AURORA__DENSE --pretrain "$CK"
# drop stale JSONs -> eval_batch recomputes only the missing CF-AURORA rows
rm -f "$RZB/json/CompletionFormer-AURORA__TEST.json" "$RZB/json/CompletionFormer-AURORA__DENSE.json"
conda activate bp 2>/dev/null || true
PYBP=/path/to/aurora-root/anaconda3/envs/bp/bin/python
$PYBP "$RZB/eval_batch.py" --jobs "$RZB/jobs_all.tsv"
$PYBP "$RZB/aggregate.py"
echo "[cf_post] DONE — CompletionFormer-AURORA rows refreshed in website + tables.tex"
