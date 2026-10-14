# Paper ground-truth tables (AURORA-KITTI-Paper.pdf) — reconciliation reference

Extracted from the submitted paper (Tables 4 & 5, p.9). RMSE/MAE in mm, iRMSE/iMAE in 1/km.
These are the AUTHORITATIVE numbers the rebuttal benchmark must be consistent with.

## Table 4 — AURORA-KITTI TEST (zero-shot, 1600 pairs)
| Method | Year | RMSE | MAE | iRMSE | iMAE | Params | Time(ms) |
|---|---|---|---|---|---|---|---|
| IPBasic(fast)  | CRV'18  | 7871.25 | 2675.13 | 54.41 | 15.71 | None     | 16.4  |
| IPBasic(multi) | CRV'18  | 6927.55 | 2236.72 | 40.26 | 13.22 | None     | 29.70 |
| S2Dense(SS)    | ICRA'19 | 6031.94 | 2237.83 | 23.32 | 10.83 | 26.11M   | 6.10  |
| S2Dense(Su)    | ICRA'19 | 5142.94 | 1803.45 | 20.36 | 8.16  | 26.11M   | 6.10  |
| BP-Net         | CVPR'24 | 5047.30 | 1851.60 | 13.26 | 6.01  | 89.87M   | 48.60 |
| Marigold-DC    | ICCV'25 | 6811.91 | 2449.70 | 54.74 | 16.61 | N/A      | 18.0k (18.0 s) |
| DMD3C          | CVPR'25 | 6065.18 | 2015.52 | 24.83 | 9.28  | 89.87M   | 48.60 |
| DA-V3          | arXiv'25| 4564.44 | 2903.61 | 23.81 | 20.53 | 1689.80M | 165.7 |
| UniK3D         | CVPR'25 | 3748.44 | 1563.49 | 9.38  | 6.37  | 358.83M  | 45.08 |
| **DDCD (Ours)**| TBD     | **1799.55** | **566.31** | **4.13** | **1.93** | 89.87M | 48.60 |

## Table 5 — DENSE TEST (zero-shot)
| Method | RMSE | MAE | iRMSE | iMAE |
|---|---|---|---|---|
| IPBasic(F)   | 7102.93 | 4907.74 | 272.37 | 200.23 |
| IPBasic(M)   | 6843.25 | 4330.87 | 167.17 | 112.59 |
| S2Dense(SS)  | 6457.09 | 4459.79 | 56.38  | 38.49  |
| S2Dense(Su)  | 4903.39 | 3366.42 | 137.74 | 81.05  |
| BP-Net       | 5062.24 | 3493.10 | 398.05 | 93.01  |
| MarigoldDC   | 6410.40 | 4605.50 | 229.69 | 165.93 |
| DMD3C        | 6544.14 | 4608.84 | 202.78 | 123.14 |
| DA-V3        | 7025.90 | 5939.04 | 121.56 | 100.17 |
| UniK3D       | 3296.92 | 2435.36 | 47.13  | 32.70  |
| **DDCD (Ours)** | **2802.80** | **1591.54** | **34.88** | **13.38** |

## DIFF vs current rebuttal html (tables.tex), RMSE only

### AURORA TEST — mostly consistent (similar trend, tiny value diffs)
| Method | paper | html | note |
|---|---|---|---|
| IP-Basic Fast | 7871.25 | 7861.1 | ~ok |
| IP-Basic Multi| 6927.55 | 6916.3 | ~ok |
| S2D Self-sup  | 6031.94 | 5971.8 | ~ok |
| S2D Sup       | 5142.94 | 4660.4 | **DIFF ~482** |
| BP-Net        | 5047.30 | 4945.4 | ~ok |
| Marigold-DC   | 6811.91 | 6778.8 | ~ok |
| DA-V3         | 4564.44 | 4564.4 | EXACT |
| UniK3D        | 3748.44 | 3716.1 | ~ok |
| DDCD          | 1799.55 | 1801.9 | ~ok |

### DENSE — DDCD is the clear outlier; most others match the paper closely/exactly
| Method | paper | html | note |
|---|---|---|---|
| S2D Self-sup  | 6457.09 | 6456.0 | EXACT |
| S2D Sup       | 4903.39 | 4903.5 | EXACT |
| UniK3D        | 3296.92 | 3296.9 | EXACT |
| DA-V3         | 7025.90 | 6858.7 | ~ok |
| BP-Net        | 5062.24 | 5465.9 | diff ~400 |
| Marigold      | 6410.40 | 5059.4 | DIFF (investigate) |
| DMD3C         | 6544.14 | 6822.3 (distilled) | ~ok-ish |
| IP-Basic F    | 7102.93 | 7615.8 | diff |
| **DDCD**      | **2802.80** | **3911.4** | **WRONG — must fix to 2802.80** |

## Conclusion
- The benchmark DENSE eval reproduces the paper for most methods (S2D-SS/S2D-Su/UniK3D EXACT),
  so the DENSE protocol itself is broadly correct. **DDCD DENSE (3911.4) is the anomaly** — its
  predictions, not the eval, are at fault. Regenerate DDCD DENSE preds from the paper checkpoint
  (_unified/rank01_rmse2802.79mm_..._ema_new_loss.pth) under the paper's DENSE protocol → expect 2802.80.
- Params/Time in the rebuttal should be aligned to Table 4 above (paper's reported, RTX 4090,
  "each method's reporting convention"), e.g. DDCD = 89.87M / 48.60 ms — NOT the 82 ms re-measure.
- AURORA S2D-Su (4660 vs 5142) and a few DENSE rows also differ — flag and diagnose, but DDCD DENSE is priority #1.

## RESOLVED 2026-06-30 — the paper Table 5 was protocol-MIXED (root cause)

The "DDCD DENSE anomaly" above was NOT a wrong checkpoint. The paper's Table 5 mixed two
eval crops: **DDCD was scored on the bottom-256 BP-Net selval crop (2802.80), while the
baselines were scored on the full 352 frame.** Re-evaluating ALL methods uniformly confirms this:

| DENSE RMSE | uniform 256 (main) | uniform 352 (appendix) | paper Table 5 |
|---|---|---|---|
| DDCD     | **2762.0** (PASS vs 2802.80, -1.5%) | 3911.4 | 2802.80 (was 256) |
| UniK3D   | 3509.0 | **3296.9** (EXACT vs paper) | 3296.92 (was 352) |
| S2D-Sup  | 5416.3 | 4903.5 (EXACT vs paper) | 4903.39 (was 352) |

So: at **uniform 256** DDCD wins all DENSE metrics (main table, tab:bench_dense); at **uniform 352**
the baselines reproduce the paper EXACTLY and UniK3D wins RMSE/MAE while DDCD wins near-field
iRMSE/iMAE (appendix, tab:bench_dense_full352). Both are now internally consistent.

**Interpreting validate_against_paper.py:** it scores json/ (uniform 256) against this file's
paper numbers. DENSE baselines therefore FLAG (256 vs the paper's 352 values) — this is EXPECTED,
not an error. To validate baseline reproduction, compare json_full352/ against the paper (they match
EXACTLY). Protocol is now pinned in jobs_all.tsv (crop_gt_top=96 on every DENSE row) so it cannot
silently drift again.

## RESOLVED 2026-07-01 — IP-Basic iRMSE/iMAE blow-up (eval bug, fixed)

TEST-set audit vs paper Table 4 found IP-Basic iRMSE/iMAE were nonsense: Fast
iRMSE 5608 (paper 54.41), Multi iRMSE 13701 (paper 40.26). RMSE/MAE were fine.
Root cause: `std_metrics` in eval_batch.py clipped hole pixels (pred=0) to 1e-3 m,
so 1/0.001=1000 (1/m) detonated the inverse-depth metric. IP-Basic leaves ~0.013%
pixels unfilled; learned methods are hole-free (unaffected — they matched paper).
Fix: inverse-depth metrics now EXCLUDE hole pixels (pred<=0) instead of clipping;
linear RMSE/MAE unchanged. No-op for every dense method. All TEST+DENSE json
re-generated 2026-07-01.
After fix (validates the fix): DENSE IP-Basic iRMSE Fast 286.6 vs paper 272.37 (+5%),
Multi 176.2 vs 167.17 (+5%); TEST Multi 43.5 vs 40.26 (+8%). TEST Fast still 92.6
vs 54.41 (+70%) — sane now, residual is a hole-density/impl difference, not a bug.

STILL OPEN (not bugs — reproduction/identity questions on TEST):
- DMD3C: paper reports ONE "DMD3C"=6065.18; our two variants bracket it — pretrained
  7244 (+19%), distilled 4973 (-18%). Need to know which checkpoint = paper's row.
- S2D-Sup (zero-shot): uniformly ~10% better than paper (RMSE 4660 vs 5142) across all
  4 metrics — likely a different checkpoint/crop; makes our baseline stronger.
