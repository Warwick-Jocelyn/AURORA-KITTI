# Depth_Models

Depth-completion code for **AURORA-KITTI: Any-Weather Depth Completion and Denoising**
(Information Fusion 139 (2027) 104830).

## What is in here — and what deliberately is not

This folder contains **only code that is ours**: the DDCD method, the evaluation and
benchmark harness, the dataset-construction tooling, and the specific adaptations that let
each public baseline run on AURORA-KITTI.

It does **not** vendor the baselines themselves. Every baseline is a public repository, so
re-cloning it at the pinned commit and dropping our files on top reproduces our environment
exactly — without redistributing thousands of files we did not write. See
[UPSTREAM.md](UPSTREAM.md) for the pinned commit of every repository.

Paths in scripts have been replaced with `/path/to/aurora-root` placeholders. Set them to
your own layout before running.

## Layout

```
Depth_Models/
├── DDCD/                 the proposed method — distillation losses, trainer, configs, dataloaders
│   ├── _bpnet_host/      BP-Net modules required on PYTHONPATH
├── benchmark/            evaluation harness + the published result tables
│   └── splits/           the frozen 1,600-frame TEST / VAL / DENSE indices
├── tools/                dataset construction: crops, resizes, split builders, validators
└── baselines/            our adaptations only, one folder per upstream repo
```

## DDCD — the method

| File | What it is |
|---|---|
| `dmd3c_distill_loss.py` | Energy-weighted SSI distillation + GT-anchored teacher normalisation — the core contribution |
| `dmd3c_distill_loss_original.py` | Pre-energy baseline loss (ablation reference) |
| `criteria.py` | SSIM/MS-SSIM/14-metric suite + `DenoiseMetrics` (ERR, SR, DCE, NRR, CPP) |
| `train.py` / `test.py` | Training and evaluation entrypoints |
| `datasets/kitti_v2clean.py` | **Headline DDCD loader** — clean Depth-Anything-V2 teacher |
| `datasets/kitti.py` | Weather RGB + clean DA-V3 teacher |
| `datasets/kitti_energyweighted.py` | Dual-teacher (clean + noisy) loader for the energy branch |
| `train_ablation_nonorm.py` | `train.py` with the `++no_teacher_norm` branch (Eq. 6 isolation) |
| `COMMANDS.md` | **Authoritative command reference** — start here to run anything |

Teacher convention: `disp/` = Depth-Anything-**V2**, `da3/` = Depth-Anything-**V3**. The
headline DDCD result uses V2 (`configs/data/KITTI_V2CLEAN.yaml`).

## Data setup

[`DATA_SETUP.md`](DATA_SETUP.md) records exactly what to download (KITTI raw, KITTI
depth-completion, AURORA-KITTI, DENSE), the expected directory layout, and how the
Depth-Anything teacher outputs are produced.

## Benchmark

`benchmark/eval_methods.py` produces the paper's tables. It reads per-method prediction
directories and the frozen split indices in `benchmark/splits/`.

Two things to know before using it:

- **Decode scales are not uniform.** `depth_scale` is `1000.0` for OMNI-DC/DENSE and both
  PromptDA directories, and `256.0` for everything else. The mapping lives in
  `eval_methods.py` and nowhere else.
- **Two crop protocols.** Plain `_DENSE` prediction directories are the bottom-256 main-table
  protocol; `_h352` directories are the full-352 appendix protocol.

`benchmark/validate_against_paper.py` is a regression gate: it parses
`paper_reference_tables.md` and fails if any aggregate RMSE drifts more than 3 % relative or
50 mm absolute from the published value. Run it after any re-evaluation.

## Checkpoints

Weights are **not** in this repository — individual files are 319–344 MB, over GitHub's
100 MB limit. They are published separately; see the repository root README for links.

| Checkpoint | Backs |
|---|---|
| `ddcd_aurora.pth` | **DDCD.** AURORA-KITTI test RMSE 1801.93 mm · DENSE RMSE 2761.97 mm |
| `ddcd_noisy_prior.pth` | Noisy-prior distillation ablation |
| `ddcd_no_teacher_norm.pth` | Teacher-normalisation ablation (Eq. 6) |
| `completionformer_aurora.pth` | CompletionFormer retrained on AURORA-KITTI |
| `bpnet_aurora.pth` | BP-Net retrained on AURORA-KITTI |

## Evaluation protocol

Two crop protocols are reported. The main tables use the bottom-256 region; the appendix
reports the full 352-row protocol. Prediction directories are named accordingly, and
`benchmark/eval_methods.py` carries the per-method decode scale for each.

DENSE provides two native rain severity levels and three fog severity levels, following the
Seeing-Through-Fog chamber protocol; severity bins follow the dataset's own discretisation.
