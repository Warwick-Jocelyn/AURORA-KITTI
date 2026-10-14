#!/usr/bin/env bash
# DENSE eval: run BOTH full-DDCD (V2 paperckpt) and no-norm(v2) through the SAME
# h352 recipe, so the no-norm-vs-full comparison is apples-to-apples, and validate
# that full-DDCD reproduces the published DDCD__DENSE.json (~2762 RMSE).
# Safely swaps the shared datas/kitti/val_selection_cropped symlink to DENSE and
# ALWAYS restores it on exit (a wrong symlink would corrupt future TEST evals).
set -u
ROOT=/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
cd "$ROOT"
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
export PYTHONPATH=$(realpath ..):
RZB=/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Experiments_Plan/results_zeroshot_benchmark

LINK=datas/kitti/val_selection_cropped
ORIG_TARGET=$(readlink "$LINK")   # val_selection_cropped_TEST
echo "[$(date '+%F %T')] original symlink target = $ORIG_TARGET"
restore() { ln -sfn "$ORIG_TARGET" "$LINK"; echo "[$(date '+%F %T')] symlink restored -> $(readlink "$LINK")"; }
trap restore EXIT

FULL_CKPT=checkpoints/_unified/rank01_rmse2802.79mm_dmd3d_bp_kitti_best_ema_new_loss.pth
NONORM_CKPT=checkpoint/DDCD_ablation_nonorm_v2/best_ema.pth
FULL_PRED=./predictions/DDCD_full_DENSE_recheck
NONORM_PRED=./predictions/DDCD_nonorm_v2_DENSE

echo "[$(date '+%F %T')] swapping symlink -> val_selection_cropped_DENSE"
ln -sfn val_selection_cropped_DENSE "$LINK"
[ "$(readlink "$LINK")" = "val_selection_cropped_DENSE" ] || { echo "SYMLINK SWAP FAILED"; exit 3; }

# Published DENSE recipe uses BP-Net KITTI mean/std (NOT the DMD3C KITTI352 values).
MEAN='[90.995,96.2278,94.3213]'
STD='[79.2382,80.5267,82.1483]'
run_infer() {  # name ckpt savedir
  local name="$1" ckpt="$2" sd="$3"
  echo "[$(date '+%F %T')] inference: $name  (ckpt=$ckpt, published mean/std)"
  $PY -u test.py gpus=[0] name="$name" \
    ++chpt="$ckpt" net=PMP num_workers=1 data=KITTI352 data.testset.mode=selval \
    ++data.mean="$MEAN" ++data.std="$STD" \
    test_batch_size=1 metric=MetricALL ++save=true ++save_dir="$sd" \
    > "runs/${name}.log" 2>&1
  local n=$(ls "$sd" 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] $name done: $n preds"
}

run_infer DDCD_full_DENSE_recheck "$FULL_CKPT" "$FULL_PRED"
run_infer DDCD_nonorm_v2_DENSE   "$NONORM_CKPT" "$NONORM_PRED"

restore; trap - EXIT   # restore symlink now; eval reads DENSE dir directly

echo "[$(date '+%F %T')] evaluating both on DENSE ..."
$PY "$RZB/eval_one.py" --name DDCD_full_DENSE_recheck --set DENSE --pred_dir "$FULL_PRED" \
    --name_map gt --pred_scale 256 --out "$RZB/json/DDCD_full_DENSE_recheck__DENSE.json"
$PY "$RZB/eval_one.py" --name DDCD_nonorm_v2 --set DENSE --pred_dir "$NONORM_PRED" \
    --name_map gt --pred_scale 256 --out "$RZB/json/DDCD_nonorm_v2__DENSE.json"

echo "===== DENSE COMPARISON (ALL group) ====="
$PY - "$RZB/json/DDCD_nonorm_v2__DENSE.json" "$RZB/json/DDCD_full_DENSE_recheck__DENSE.json" "$RZB/json/DDCD__DENSE.json" <<'PYEOF'
import json, sys
nn  = json.load(open(sys.argv[1]))["groups"]["ALL"]
fl  = json.load(open(sys.argv[2]))["groups"]["ALL"]
pub = json.load(open(sys.argv[3]))["groups"]["ALL"]
ks = ["RMSE","MAE","iRMSE","iMAE"]
print(f"{'metric':7} {'no-norm(v2)':>13} {'full-recheck':>13} {'published DDCD':>15} {'delta(nn-full)':>15}")
for k in ks:
    print(f"{k:7} {nn[k]:13.3f} {fl[k]:13.3f} {pub[k]:15.3f} {nn[k]-fl[k]:+15.3f}")
print("\nValidation: full-recheck RMSE should ~= published DDCD RMSE (2762) => recipe correct.")
print("Interpretation: positive delta = no-norm WORSE on real DENSE => normalization helps transfer.")
PYEOF
echo "[$(date '+%F %T')] DENSE ALL DONE"
