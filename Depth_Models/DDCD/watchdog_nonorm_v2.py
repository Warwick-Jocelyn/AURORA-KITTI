#!/usr/bin/env python
"""
Deterministic early-stopping watchdog for run `DDCD_ablation_nonorm_v2`.

This is a standalone monitoring script -- it does NOT modify, import, or
retrain anything. It only:
  1. tails runs/DDCD_ablation_nonorm/train.log for `[VAL] Epoch N primary
     metric=...` lines (lower RMSE = better),
  2. tracks the best epoch and how many epochs since the last improvement,
  3. if no_improve patience is exceeded (with a minimum-epoch guard), sends
     SIGTERM to ONLY the main train_ablation_nonorm.py pid (never pkill /
     pattern kill -- other GPU jobs may be running on this machine),
  4. writes a status line every poll and a final marker file summarizing
     why/when it stopped (or that the run finished naturally).

Run with nohup, e.g.:
    nohup python -u watchdog_nonorm.py > runs/DDCD_ablation_nonorm/watchdog_stdout.log 2>&1 &
"""

import os
import re
import signal
import subprocess
import time
from datetime import datetime

WORKDIR = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C"
RUN_NAME = "DDCD_ablation_nonorm_v2"
LOG_PATH = os.path.join(WORKDIR, "runs", RUN_NAME, "train.log")
WATCHDOG_LOG = os.path.join(WORKDIR, "runs", RUN_NAME, "watchdog.log")
MARKER_STOP = os.path.join(WORKDIR, "runs", RUN_NAME, "EARLYSTOP.marker")
MARKER_DONE = os.path.join(WORKDIR, "runs", RUN_NAME, "COMPLETED.marker")
PROC_PATTERN = "train_ablation_nonorm.py"

POLL_SECONDS = 300
MIN_EPOCH_GUARD = 6     # do not early-stop before this epoch has been reached
PATIENCE = 5            # epochs since best allowed before stopping
NATURAL_END_EPOCH = 29  # nepoch=30 -> epochs 0..29, 29 is the last one

VAL_RE = re.compile(
    r"\[VAL\]\s+Epoch\s+(\d+)\s+primary metric=([-\w\.]+)\s+\(prev best=([-\w\.]+)\)"
)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def find_main_pid():
    """Return the smallest pid matching PROC_PATTERN (the main process;
    larger pids are its dataloader worker children), or None if not running.
    """
    try:
        out = subprocess.check_output(["pgrep", "-f", PROC_PATTERN]).decode()
    except subprocess.CalledProcessError:
        return None
    pids = [int(p) for p in out.split() if p.strip()]
    if not pids:
        return None
    return min(pids)


def parse_val_curve():
    """Read train.log, normalize \\r-> \\n (tqdm rewrites lines with \\r),
    and parse all [VAL] Epoch N primary metric=X lines into {epoch: value}.
    Later occurrences of the same epoch overwrite earlier ones (in case of
    a resumed/re-logged epoch).
    """
    if not os.path.exists(LOG_PATH):
        return {}
    with open(LOG_PATH, "r", errors="ignore") as f:
        raw = f.read()
    text = raw.replace("\r", "\n")
    curve = {}
    for m in VAL_RE.finditer(text):
        epoch = int(m.group(1))
        val_str = m.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue  # e.g. "nan"/"inf" strings that don't parse cleanly
        if not (val == val):  # NaN check
            continue
        curve[epoch] = val
    return curve


def write_marker(path, **fields):
    with open(path, "w") as f:
        for k, v in fields.items():
            f.write(f"{k}={v}\n")


def main():
    log(f"Watchdog started. workdir={WORKDIR} log={LOG_PATH} "
        f"poll={POLL_SECONDS}s min_epoch_guard={MIN_EPOCH_GUARD} patience={PATIENCE}")

    while True:
        if not os.path.exists(LOG_PATH):
            log("train.log does not exist yet, waiting...")
            time.sleep(POLL_SECONDS)
            continue

        main_pid = find_main_pid()
        if main_pid is None:
            log(f"No '{PROC_PATTERN}' process found -- run has already ended "
                "(or was stopped outside this watchdog). Exiting watchdog.")
            return

        curve = parse_val_curve()
        if not curve:
            log(f"main_pid={main_pid} alive; no [VAL] lines parsed yet "
                "(still in an early epoch / first validation not reached).")
            time.sleep(POLL_SECONDS)
            continue

        cur_epoch = max(curve.keys())
        cur_val = curve[cur_epoch]
        best_epoch = min(curve, key=lambda e: curve[e])
        best_val = curve[best_epoch]
        no_improve = cur_epoch - best_epoch

        log(f"main_pid={main_pid} cur_epoch={cur_epoch} cur_val={cur_val:.6f} "
            f"best_epoch={best_epoch} best_val={best_val:.6f} no_improve={no_improve} "
            f"epochs_seen={sorted(curve.keys())}")

        if cur_epoch >= MIN_EPOCH_GUARD and no_improve >= PATIENCE:
            log(f"EARLY STOP condition met: cur_epoch={cur_epoch} >= {MIN_EPOCH_GUARD} "
                f"and no_improve={no_improve} >= {PATIENCE}. "
                f"Sending SIGTERM to main pid {main_pid} only.")
            try:
                os.kill(main_pid, signal.SIGTERM)
                kill_ok = True
            except ProcessLookupError:
                log(f"pid {main_pid} already gone before SIGTERM could be sent.")
                kill_ok = False
            write_marker(
                MARKER_STOP,
                early_stopped=1,
                best_epoch=best_epoch,
                best_val=best_val,
                cur_epoch=cur_epoch,
                no_improve=no_improve,
                killed_pid=main_pid,
                kill_sent=kill_ok,
                timestamp=datetime.now().isoformat(),
            )
            log(f"Wrote {MARKER_STOP}. Watchdog exiting.")
            return

        if cur_epoch >= NATURAL_END_EPOCH:
            log(f"Reached natural end epoch (cur_epoch={cur_epoch} >= {NATURAL_END_EPOCH}). "
                "Writing completed marker and exiting (no kill needed).")
            write_marker(
                MARKER_DONE,
                completed=1,
                best_epoch=best_epoch,
                best_val=best_val,
                cur_epoch=cur_epoch,
                timestamp=datetime.now().isoformat(),
            )
            return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
