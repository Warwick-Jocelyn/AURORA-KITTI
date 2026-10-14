#!/bin/bash
# Full chain: proper retrain -> inference TEST+DENSE -> re-eval -> refresh website/tex.
# Self-contained; survives VS Code / SSH close (run under tmux). Logs to cf_full.log.
# Guarded: inference/eval only run if training succeeded AND a checkpoint exists.
LOG=/path/to/aurora-root/Code/CompletionFormer/cf_full.log
exec > >(tee -a "$LOG") 2>&1
CFE=/path/to/aurora-root/Code/CompletionFormer/experiments_aurora_full
echo "==== CF FULL CHAIN START $(date) ===="
bash /path/to/aurora-root/Code/CompletionFormer/run_cf_full.sh
rc=$?
CK=$(ls -t "$CFE"/model_*.pt 2>/dev/null | head -1)
if [ $rc -ne 0 ] || [ -z "$CK" ]; then
    echo "!!!! TRAINING FAILED (rc=$rc, ckpt='$CK') — ABORTING chain, leaving existing results untouched"
    exit 1
fi
echo "==== TRAINING DONE (ckpt=$CK), starting inference+eval $(date) ===="
bash /path/to/aurora-root/Code/CompletionFormer/cf_post.sh "$CK"
echo "==== CF FULL CHAIN COMPLETE $(date) ===="
touch /path/to/aurora-root/Code/CompletionFormer/CF_FULL_DONE
