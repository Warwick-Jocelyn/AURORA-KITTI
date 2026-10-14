#!/usr/bin/env bash
# Orchestrator: wait for the v2 no-norm TEST352 inference to finish (1600 preds),
# then run eval_one.py on the TEST split and print the comparison vs the
# same-pipeline full-DDCD baseline. Safe to run detached; clamonacc may delay the
# inference start by minutes (cold on-access scan) — we just wait it out.
set -u
cd /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
PRED=./predictions/DDCD_ablation_nonorm_v2_TEST352
RZB=/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Experiments_Plan/results_zeroshot_benchmark
OUT=$RZB/json/DDCD_ablation_nonorm_v2__TEST.json
INFER_PID=274201

echo "[$(date '+%F %T')] waiting for inference (pid $INFER_PID) to produce 1600 preds ..."
# Wait until 1600 preds exist AND the inference process has exited (so all files flushed).
while true; do
  n=$(ls "$PRED" 2>/dev/null | wc -l)
  alive=$(ps -p $INFER_PID -o pid= 2>/dev/null | wc -l)
  if [ "$n" -ge 1600 ] && [ "$alive" -eq 0 ]; then break; fi
  if [ "$alive" -eq 0 ] && [ "$n" -lt 1600 ]; then
    echo "[$(date '+%F %T')] ERROR: inference exited with only $n preds"; exit 2
  fi
  sleep 5
done
echo "[$(date '+%F %T')] inference done: $(ls "$PRED" | wc -l) preds. Running eval ..."

$PY "$RZB/eval_one.py" --name DDCD_ablation_nonorm_v2 --set TEST \
  --pred_dir "$PRED" --name_map gt --pred_scale 256 --out "$OUT"
echo "[$(date '+%F %T')] eval done -> $OUT"

echo "===== COMPARISON (ALL group) ====="
$PY - "$OUT" "$RZB/json/DDCD_full_samepipe__TEST.json" <<'PYEOF'
import json, sys
nn = json.load(open(sys.argv[1]))["groups"]["ALL"]
fl = json.load(open(sys.argv[2]))["groups"]["ALL"]
keys = ["RMSE","MAE","iRMSE","iMAE"]
print(f"{'metric':8} {'no-norm(v2)':>14} {'full-DDCD':>12} {'delta(nn-full)':>16}")
for k in keys:
    a,b = nn[k], fl[k]
    print(f"{k:8} {a:14.4f} {b:12.4f} {a-b:+16.4f}")
print("\nInterpretation: positive delta = no-norm WORSE (higher error) => normalization helps.")
PYEOF
echo "[$(date '+%F %T')] ALL DONE"
