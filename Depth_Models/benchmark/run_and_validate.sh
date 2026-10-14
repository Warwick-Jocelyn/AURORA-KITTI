#!/bin/bash
# run_and_validate.sh — aggregate tables then gate against paper ground-truth.
# Called after every eval_batch.py run to ensure silent divergence can't slip through.
#
# Exit codes:
#   0 — aggregate succeeded + all paper rows PASS
#   1 — aggregate succeeded + one or more rows FLAGGED
#   2 — aggregate itself failed

set -e
PYTHON="/path/to/aurora-root/anaconda3/envs/bp/bin/python"
RESULTS="$(cd "$(dirname "$0")" && pwd)"
HTML="$RESULTS/results.html"
DATE=$(date '+%Y-%m-%d %H:%M BST')

echo "========================================"
echo "  run_and_validate.sh  ($DATE)"
echo "========================================"

# ── 1. Aggregate ─────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Running aggregate.py ..."
cd "$RESULTS"
if ! $PYTHON aggregate.py; then
    echo "FATAL: aggregate.py exited non-zero." >&2
    exit 2
fi
echo "      aggregate.py done -> results.html, tables.tex, results.md"

# ── 2. Validate ──────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Running validate_against_paper.py ..."
VALIDATE_OUT=$($PYTHON validate_against_paper.py 2>&1)
VALIDATE_EXIT=$?
echo "$VALIDATE_OUT"

# Parse the VALIDATION_SUMMARY line emitted by the script
SUMMARY_LINE=$(echo "$VALIDATE_OUT" | grep "^VALIDATION_SUMMARY" | tail -1)
N_PASS=$(echo "$SUMMARY_LINE" | grep -oP 'n_pass=\K[0-9]+' || echo "?")
N_FLAG=$(echo "$SUMMARY_LINE" | grep -oP 'n_flag=\K[0-9]+' || echo "?")
N_MISS=$(echo "$SUMMARY_LINE" | grep -oP 'n_miss=\K[0-9]+' || echo "?")
VAL_DATE=$(echo "$SUMMARY_LINE" | grep -oP 'date=\K[0-9-]+' || echo "$DATE")

# ── 3. Inject validation banner into results.html ────────────────────────────
if [ -f "$HTML" ]; then
    if [ "$VALIDATE_EXIT" -eq 0 ]; then
        BANNER_COLOR="#d4edda"
        BANNER_ICON="&#10003;"
        BANNER_TEXT="Validated against paper Tables 4/5 on ${VAL_DATE}: <b>${N_PASS} rows within tolerance, ${N_FLAG} flagged</b>. All paper-reference methods PASS."
    else
        BANNER_COLOR="#fff3cd"
        BANNER_ICON="&#9888;"
        BANNER_TEXT="Validated against paper Tables 4/5 on ${VAL_DATE}: <b>${N_PASS} rows PASS, ${N_FLAG} FLAGGED</b>. See validate_against_paper.py output for details."
    fi

    BANNER_HTML="<div style='background:${BANNER_COLOR};border:1px solid #c3c3a0;border-radius:4px;padding:8px 14px;margin:8px 0;font-size:0.92em;'>${BANNER_ICON} ${BANNER_TEXT}</div>"

    # Inject after the first <h1> tag (aggregate.py always produces one)
    python3 - <<PYEOF
import re
with open("$HTML") as f:
    content = f.read()
banner = """$BANNER_HTML"""
# Insert after the first </h1>
new_content = re.sub(r'(</h1>)', r'\1\n' + banner, content, count=1)
with open("$HTML", "w") as f:
    f.write(new_content)
print("      Validation banner injected into results.html")
PYEOF
fi

# ── 4. Final banner ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
if [ "$VALIDATE_EXIT" -eq 0 ]; then
    echo "  RESULT: ALL PASS (${N_PASS} rows within 3%/50mm tolerance)"
    echo "  results.html updated with green validation banner."
else
    echo "  !! WARNING: ${N_FLAG} row(s) FLAGGED !!"
    echo "  results.html updated with yellow validation banner."
    echo "  See validate_against_paper.py output above for details."
fi
echo "========================================"

exit "$VALIDATE_EXIT"
