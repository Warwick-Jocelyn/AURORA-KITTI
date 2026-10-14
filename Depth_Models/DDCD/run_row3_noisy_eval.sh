#!/usr/bin/env bash
set -u
ROOT=/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
cd "$ROOT"; export PYTHONPATH=$(realpath ..):
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
RZB=/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Experiments_Plan/results_zeroshot_benchmark
CKPT=checkpoints/_unified/rank02_rmse3066.92mm_dmd3d_bp_kitti_noisy_disp_best_ema_noisy_dis.pth
PRED=./predictions/DDCD_noisy_disp_TEST352
# symlink must be TEST for AURORA-KITTI TEST
[ "$(readlink datas/kitti/val_selection_cropped)" = "val_selection_cropped_TEST" ] || { echo "SYMLINK NOT TEST -- abort"; exit 3; }
echo "[$(date '+%F %T')] infer row3 noisy-prior distill ..."
$PY -u test.py gpus=[0] name=DDCD_noisy_disp_TEST352 ++chpt="$CKPT" net=PMP num_workers=1 \
  data=KITTI352 data.testset.mode=selval test_batch_size=1 metric=MetricALL ++save=true ++save_dir="$PRED" \
  > runs/DDCD_noisy_disp_TEST352.log 2>&1
echo "[$(date '+%F %T')] preds: $(ls "$PRED" | wc -l)"
$PY "$RZB/eval_one.py" --name DDCD_noisy_disp --set TEST --pred_dir "$PRED" --name_map gt --pred_scale 256 \
  --out "$RZB/json/DDCD_noisy_disp__TEST.json"
echo "ROW3 DONE (expect RMSE ~1956 to confirm identity)"
