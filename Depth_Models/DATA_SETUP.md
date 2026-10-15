# Data setup — obtaining and laying out the source data

Everything AURORA-KITTI is built from is either publicly downloadable or published by us. This
file records exactly what to fetch, where it goes, and how it was used, so no local copy needs
to be kept.

---

## 1. KITTI raw (public)

AURORA-KITTI derives from **156 KITTI raw drives** across 5 calibration days. The full drive
list is in [`benchmark/KITTI_SOURCE_DRIVES.md`](benchmark/KITTI_SOURCE_DRIVES.md).

**Source:** <https://www.cvlibs.net/datasets/kitti/raw_data.php> (licence: CC BY-NC-SA 3.0)

The raw-data page offers a per-drive download script. To fetch only the drives we use:

```bash
# one zip per drive, from the drive list
while read -r drive; do
  [ -z "$drive" ] && continue
  base="${drive%_sync}"
  wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${base}/${drive}.zip"
  unzip -q "${drive}.zip" -d raw/
done < drives.txt

# calibration, one per day
for day in 2011_09_26 2011_09_28 2011_09_29 2011_09_30 2011_10_03; do
  wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${day}_calib.zip"
  unzip -q "${day}_calib.zip" -d raw/
done
```

`drives.txt` is the fenced block at the end of `KITTI_SOURCE_DRIVES.md`.

## 2. KITTI depth completion (public)

```bash
B=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth
wget $B'_annotated.zip' $B'_velodyne.zip' $B'_selection.zip'
```

**Source:** <https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion>

## 3. AURORA-KITTI itself (ours)

```bash
hf download JocelynW/AURORA-KITTI --repo-type dataset --local-dir aurora-kitti
```

The 1,600-frame benchmark the paper's tables are computed on:

```bash
hf download JocelynW/AURORA-KITTI --repo-type dataset \
    --include "benchmark_1600/*" --local-dir aurora-kitti
```

## 4. DENSE / Seeing Through Fog (third party, gated)

Registration required: <https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/>
Used only for the real-world cross-dataset evaluation.

---

## Expected layout

The Hydra configs resolve data under `datas/` relative to the DDCD working directory
(`configs/data/*.yaml` → `path: datas/kitti`):

```
datas/
├── kitti/
│   ├── raw/<date>/<drive>/image_0{2,3}/data/     KITTI raw images
│   ├── data_depth_annotated/{train,val}/         dense GT
│   ├── data_depth_velodyne/{train,val}/          sparse LiDAR
│   ├── {train,val,test}.csv                      AURORA-KITTI weather annotations
│   ├── val_selection_cropped_TEST/               1,600-frame AURORA harness
│   ├── val_selection_cropped_DENSE/              1,600-frame DENSE harness
│   └── val_selection_cropped_VAL/                1,600-frame val harness
└── kitti_depth_DC/
    └── raw/<date>/<drive>/image_0{2,3}/
        ├── disp/      Depth-Anything-V2 teacher disparity   <- used by the published model
        └── da3/       Depth-Anything-V3 teacher disparity   <- ablation only
```

The harnesses are rebuilt with `tools/datas/kitti/val_selection_cropped_TEST/Tools/prune_by_csv.py`
and the crop utilities in `tools/datas/Tools/`.

## Teacher outputs

The frozen teacher's predictions are precomputed per frame rather than run during training.

| Folder | Teacher | Used by |
|---|---|---|
| `disp/` | Depth-Anything-**V2** ViT-L | the published DDCD (`configs/data/KITTI_V2CLEAN.yaml`) |
| `da3/` | Depth-Anything-**V3** | the V2-vs-V3 ablation only |

Regenerate with the batch runners in `baselines/Depth-Anything-V2/run_batch.py` and
`baselines/Depth-Anything-3/run_batch_da3_kitti_raw.py`, using the public weights listed in
[`UPSTREAM.md`](UPSTREAM.md). Over ~93k frames this takes many GPU-hours, so the V2 outputs
(9.66 GiB) are also kept in a private archive.

## Provenance of what was removed locally

| Removed | Size | Recoverable from |
|---|---:|---|
| `kitti_depth/` | 97 GB | §1 + §2 |
| `DMD3C/datas/kitti_depth_DC/` | 121 GB | §1 + §2, teacher outputs regenerated or restored from archive |
| `DMD3C/datas/kitti/raw/` | 28 GB | §1 |
| `DMD3C/datas/Backup/` | 41 GB | superseded pre-crop copies of the harnesses |
| `lidar_compound_noise_models/raw_lidar/` | 44 GB | §1 (KITTI raw Velodyne) |
| `rain-rendering/` bulk | 40 GB | renders superseded by the published frames; masks archived privately |
