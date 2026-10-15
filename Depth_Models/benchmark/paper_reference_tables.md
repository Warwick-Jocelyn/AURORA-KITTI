# Published reference tables

Values as published in *AURORA-KITTI: Any-weather depth completion and denoising*,
Information Fusion 139 (2027) 104830.

`validate_against_paper.py` checks re-evaluated results against these numbers.

RMSE and MAE in mm; iRMSE and iMAE in 1/km. All methods except DDCD are evaluated zero-shot
from publicly released checkpoints; DDCD is trained in-domain.

## Table 5 — AURORA-KITTI test set

| Method | Year | RMSE | MAE | iRMSE | iMAE | ERR | DCE | CPP | Params | Time (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| IP-Basic (fast) | CRV'18 | 7861.15 | 2650.48 | 92.55 | 31.64 | -0.001 | 2416.93 | 1810.5 | — | 16.4 |
| IP-Basic (multi) | CRV'18 | 6916.28 | 2209.20 | 43.49 | 14.34 | 0.094 | 2103.64 | 1998.3 | — | 29.70 |
| S2Dense (SS) | ICRA'19 | 5971.83 | 2221.87 | 23.17 | 10.74 | 0.051 | 1850.15 | 1860.9 | 26.1M | 6.1 |
| S2Dense (Su) | ICRA'19 | 4660.44 | 1610.71 | 18.52 | 7.38 | 0.252 | 1609.60 | 1082.0 | 26.1M | 6.1 |
| NLSPN | ECCV'20 | 4013.23 | 1293.62 | 19.86 | 5.15 | 0.071 | 1375.59 | 1053.8 | 26.2M | 40.2 |
| CompletionFormer | CVPR'23 | 3961.99 | 1265.66 | 14.15 | 3.77 | 0.164 | 1315.77 | 1029.6 | 83.5M | 73.4 |
| BP-Net | CVPR'24 | 4945.43 | 1818.58 | 12.32 | 5.73 | 0.348 | 1466.38 | 1663.8 | 89.9M | 48.6 |
| Marigold-DC | ICCV'25 | 6778.75 | 2441.68 | 54.67 | 16.58 | 0.147 | 2106.19 | 1843.7 | — | 34000 |
| OMNI-DC | ICCV'25 | 6225.81 | 2080.96 | 42.34 | 15.05 | 0.010 | 2089.78 | 1415.7 | 420.7M | 320 |
| DMD3C | CVPR'25 | 7244.44 | 2638.07 | 42.65 | 15.86 | 0.003 | 2241.20 | 1826.3 | 89.87M | 48.60 |
| UniK3D | CVPR'25 | 3716.06 | 1552.24 | 9.37 | 6.36 | 0.744 | 683.03 | 3136.4 | 358.8M | 45.1 |
| DA-V3 | ICLR'26 | 4564.44 | 2903.61 | 23.81 | 20.53 | 0.683 | 888.20 | 4315.2 | 1689.8M | 165.7 |
| **DDCD (ours)** | 2026 | **1801.93** | **570.36** | **4.25** | **1.96** | 0.826 | 474.41 | 835.1 | 89.87M | 48.60 |

## Table 6 — DENSE test set (real-world)

| Method | RMSE | MAE | iRMSE | iMAE |
|---|---|---|---|---|
| IP-Basic (fast) | 7830.63 | 5582.13 | 286.58 | 217.22 |
| IP-Basic (multi) | 7346.78 | 5172.20 | 176.23 | 116.17 |
| S2Dense (SS) | 5859.08 | 3916.45 | 54.91 | 34.76 |
| S2Dense (Su) | 5416.31 | 3671.62 | 99.12 | 64.14 |
| NLSPN | 4916.51 | 3223.92 | 71.12 | 45.13 |
| CompletionFormer | 4754.04 | 3234.06 | 50.43 | 28.58 |
| BP-Net | 4946.62 | 3561.48 | 177.36 | 85.51 |
| Marigold-DC | 5364.58 | 3882.82 | 130.38 | 92.30 |
| OMNI-DC | 5752.49 | 4031.82 | 189.41 | 140.97 |
| DMD3C | 6042.42 | 4223.60 | 184.72 | 130.57 |
| DA-V3 | 7778.43 | 6118.93 | 120.46 | 83.00 |
| UniK3D | 3509.04 | 2485.10 | 49.06 | 30.77 |
| **DDCD (ours)** | **2761.97** | **1569.14** | **34.82** | **13.24** |

## Denoising metrics

ERR (higher is better) measures correction of weather-corrupted returns; DCE (lower is better)
measures dropout-region recovery; CPP (lower is better) measures preservation of reliable
returns. ERR is dimensionless; DCE and CPP are in millimetres. Definitions are in Section 3.6
of the paper.

DENSE provides no paired clean LiDAR, so the denoising metrics are reported on AURORA-KITTI
only.

## Evaluation protocol

**Both benchmarks are evaluated on a fixed 1,600-frame subset, not the full test split.**
The frame indices are `benchmark/splits/test_small.csv` and `dense_small.csv`. The main tables use the bottom-256 crop region; the appendix
reports the full 352-row protocol.
