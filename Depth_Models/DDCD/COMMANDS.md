# DMD3C — Commands Reference

Single source of truth for training, evaluation, and dataset prep on this
project. Replaces the old paste-ups under `logs_txt/` and `cmd.txt`.

> All commands assume:
> ```bash
> conda activate DMD3C
> cd /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
> export PYTHONPATH=$(realpath ..):$PYTHONPATH
> ```
> Hydra is configured with `chdir: false` so all paths below are relative
> to this directory. Outputs land in `outputs/<date>/<time>/`, runs land in
> `runs/<name>/`, checkpoints land in `checkpoint/<name>/best_ema.pth`.

---

## 1. Project layout (after cleanup)

```
BP-Net/
├── DMD3C/                          # active code
│   ├── train.py                    # main training entrypoint (W&B-aware)
│   ├── test.py                     # selval/selvar evaluation entrypoint
│   ├── eval_all_ckpts.py           # batch-evaluate every checkpoint on DENSE
│   ├── COMMANDS.md                 # this file
│   ├── configs/                    # Hydra configs (config / data / net / ...)
│   ├── checkpoints/                # original-paper + named checkpoints
│   │   ├── pretrained_mixed_singleview_256.pth   # paper init
│   │   └── DMD3D_BP_KITTI/                       # named checkpoints
│   └── checkpoint/                 # training-output checkpoints (best_ema.pth)
│       ├── BP_KITTI_noisy_disp/                  # paper distillation init
│       └── DMD3D_*/                              # per-experiment EMA bests
├── checkpoints/BP_KITTI/           # external BP-Net baseline weights
└── DMD3C_original/                 # legacy backup (REDUNDANT, candidate for removal)
```

Datasets live in `DMD3C/datas/kitti/`:
- `data_depth_annotated/{train,val}` + `data_depth_velodyne/{train,val}` + `raw/`  → train
- `val_selection_cropped_DENSE/` → DENSE val (target benchmark)
- `val_selection_cropped` → symlink to whichever val variant is being used
  (currently → `val_selection_cropped_DENSE`)

---

## 2. Common Hydra knobs

| key                  | default                  | meaning |
|----------------------|--------------------------|---------|
| `name=`              | `KITTI-AURORA`           | run name; controls `runs/<name>/` and `checkpoint/<name>/` |
| `++chpt=`            | (unset)                  | resume / init from this `.pth` (must contain `{'net': state_dict}`) |
| `gpus=[0]`           | `[0]`                    | GPU list. Multi-GPU = use `torchrun` and pass `gpus=[0,1,2,3]` |
| `data=KITTI`         |                          | choose `configs/data/<name>.yaml` |
| `data.testset.mode`  | `selval`                 | `train` / `val` / `selval` / `test` |
| `metric=MetricALL`   | `RMSE`                   | RMSE only, or full MetricALL (RMSE/MAE/iRMSE/iMAE/REL/δ/PSNR/SSIM/MS-SSIM) |
| `++net.sbn=false`    | false                    | enable SyncBN (multi-GPU only) |
| `lr=`                | `1e-3`                   | base LR |
| `nepoch=`            | `30`                     | total epochs |
| `test_epoch=`        | `25`                     | run test() every N epochs |
| `train_batch_size=`  | `2`                      | per-GPU |
| `test_batch_size=`   | `2`                      | per-GPU |
| `++lambda_distill=`  | `0.1`                    | weight on SSI distillation loss |
| `++lambda_grad=`     | `0.05`                   | weight on multi-scale residual gradient loss |
| `++max_train_images=`| unset (= full)           | int / `full` — cap train sample count for debug |
| `++max_val_images=`  | unset (= full)           | int / `full` — cap val sample count for debug |
| `++use_wandb=`       | true                     | toggle W&B logging |
| `++save=`            | true (test.py)           | save 16-bit predictions when running `test.py` |

---

## 3. Training

### 3.1 Standard run (full data, single GPU, W&B on)

```bash
nohup python -u train.py \
  gpus=[0] num_workers=4 name=DMD3D_BP_KITTI_v4 \
  ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
  net=PMP data=KITTI \
  lr=5e-4 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=30 test_epoch=1 ++net.sbn=false metric=MetricALL \
  ++lambda_distill=0.1 ++lambda_grad=0.05 \
  ++max_train_images=full ++max_val_images=1600 \
  ++use_wandb=true \
  > runs/DMD3D_BP_KITTI_v4/train.log 2>&1 &
```

### 3.2 Debug run (smoke test, no W&B)

```bash
python -u train.py \
  gpus=[0] num_workers=2 name=DBG_$(date +%H%M) \
  ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
  net=PMP data=KITTI \
  lr=5e-4 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=2 test_epoch=1 ++net.sbn=false metric=MetricALL \
  ++lambda_distill=0.1 ++lambda_grad=0.05 \
  ++max_train_images=50 ++max_val_images=8 \
  ++use_wandb=false
```

### 3.3 Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 --master_port=4321 train.py \
  gpus=[0,1,2,3] num_workers=4 name=DMD3D_DDP \
  ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
  net=PMP data=KITTI \
  lr=1e-3 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=30 test_epoch=1 ++net.sbn=true metric=MetricALL \
  ++lambda_distill=0.1 ++lambda_grad=0.05
```

### 3.4 Resume / fine-tune from a previous EMA

```bash
nohup python -u train.py \
  gpus=[0] num_workers=4 name=DMD3D_BP_KITTI_resume \
  ++chpt=checkpoint/DMD3D_BP_KITTI_v4/best_ema.pth \
  net=PMP data=KITTI \
  lr=2e-4 train_batch_size=2 test_batch_size=2 \
  sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
  nepoch=10 test_epoch=1 ++net.sbn=false metric=MetricALL \
  ++lambda_distill=0.08 ++lambda_grad=0.03 \
  ++max_train_images=full ++max_val_images=full \
  ++use_wandb=true \
  > runs/DMD3D_BP_KITTI_resume/train.log 2>&1 &
```

---

## 4. Evaluation

### 4.1 Single checkpoint on DENSE val

The `val_selection_cropped` symlink decides which val you hit. To target
DENSE specifically:
```bash
ln -sfn val_selection_cropped_DENSE datas/kitti/val_selection_cropped
```

Then:
```bash
python test.py \
  gpus=[0] num_workers=1 name=EVAL_DENSE \
  ++chpt=checkpoints/DMD3D_BP_KITTI/best_ema_for_val.pth \
  net=PMP data=KITTI \
  data.testset.mode=selval test_batch_size=2 \
  metric=MetricALL ++save=false
```

To save 16-bit prediction PNGs (`pred*256`):
```bash
python test.py \
  gpus=[0] name=EVAL_DENSE_save \
  ++chpt=checkpoints/DMD3D_BP_KITTI/best_ema_for_val.pth \
  net=PMP num_workers=1 data=KITTI \
  data.testset.mode=selval test_batch_size=1 \
  metric=MetricALL ++save=true \
  ++save_dir=./predictions/EVAL_DENSE_save
```

### 4.2 Eval ALL checkpoints in one go

`eval_all_ckpts.py` walks `CKPT_LIST` (12 unique checkpoints, deduplicated by
MD5) and writes one JSON line per checkpoint to `ckpt_eval_results.jsonl`:

```bash
python eval_all_ckpts.py 2>&1 | tee eval_all.log
```

To swap which val set is being benchmarked, point the symlink at
`val_selection_cropped_VAL` or `val_selection_cropped_TEST` first.

### 4.3 Vanilla KITTI selval (test set submission style)

```bash
python test.py \
  gpus=[0] name=KITTI_selval \
  ++chpt=checkpoint/<exp_name>/best_ema.pth \
  net=PMP num_workers=1 data=KITTI \
  data.testset.mode=selval test_batch_size=1 \
  metric=MetricALL ++save=true \
  ++save_dir=./predictions/KITTI_selval
```

---

## 5. Dataset prep

### 5.1 RGB mean/std for a new split

```bash
python datas/kitti/compute_mean_std_dense.py \
  --root datas/kitti/val_selection_cropped_DENSE/image
# → paste into configs/data/<NAME>.yaml
```

### 5.2 Resize 16-bit sparse LiDAR / calib (DENSE prep)

```bash
python datas/kitti/resize_16bit_sparse_lidar.py
python datas/kitti/resize_calib.py
```

### 5.3 Sync to HPC

```bash
rsync -av \
  /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/<exp_name> \
  user@hpc-login:/path/to/hpc-home/code/DMD3C/BP-Net/DMD3C/checkpoint
```

---

## 6. Checkpoint registry

See `checkpoints/REGISTRY.md` for the per-checkpoint DENSE-RMSE leaderboard
(produced by `eval_all_ckpts.py`).

Two checkpoints have **frozen names** because they come from upstream papers
and external code expects these exact filenames:

- `checkpoints/pretrained_mixed_singleview_256.pth` — PMP backbone init
- `checkpoint/BP_KITTI_noisy_disp/dmd3c_distillation_depth_anything_v2.pth` — DMD3C distill init

All other checkpoints have been renamed to encode their RMSE rank on the
DENSE val set so that the filesystem listing acts as a leaderboard.
