#!/bin/bash
# Parallel unified-eval runner. Reads a TSV (name<TAB>set<TAB>pred_dir<TAB>name_map<TAB>scale),
# runs eval_one.py for each row, logs per-job, writes json/<name>__<set>.json.
# Usage: run_eval.sh <jobs.tsv> [parallelism]
set -u
RZB=/path/to/aurora-root/PhD_Project_2025/Experiments_Plan/results_zeroshot_benchmark
PY=/path/to/aurora-root/anaconda3/envs/DMD3C/bin/python
JOBS="${1:-$RZB/jobs_existing.tsv}"
P="${2:-5}"
cd "$RZB" || exit 1
mkdir -p json logs
run_one() {
  IFS=$'\t' read -r name set pdir nmap scale <<< "$1"
  [ -z "$name" ] && return 0
  echo ">>> $name / $set  ($(date +%H:%M:%S))"
  "$PY" "$RZB/eval_one.py" --name "$name" --set "$set" --pred_dir "$pdir" \
        --name_map "$nmap" --pred_scale "$scale" \
        --out "$RZB/json/${name}__${set}.json" \
        > "$RZB/logs/eval_${name}__${set}.log" 2>&1 \
    && echo "    OK $name/$set" || echo "    FAIL $name/$set (see logs/eval_${name}__${set}.log)"
}
export -f run_one; export RZB PY
grep -vE '^\s*$' "$JOBS" | parallel -j "$P" run_one {} 2>/dev/null \
  || { echo "[no GNU parallel; falling back to xargs -P$P]"; \
       grep -vE '^\s*$' "$JOBS" | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$@"' _ {}; }
echo "ALL EVAL JOBS DONE $(date)"
