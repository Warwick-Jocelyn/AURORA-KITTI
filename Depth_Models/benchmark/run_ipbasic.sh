#!/bin/bash
RZB=/path/to/aurora-root/PhD_Project_2025/Information_Fusion_rebuttal/Experiments_Plan/results_zeroshot_benchmark
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
cd "$RZB" || exit 1
for fill in fast multiscale; do
  for set in TEST DENSE; do
    echo "=== IPBasic-$fill / $set $(date +%H:%M:%S) ==="
    "$PY" ipbasic_infer.py --set $set --fill $fill --out "$RZB/preds/IPBasic-${fill}__${set}"
  done
done
echo "IPBASIC ALL DONE $(date)"
