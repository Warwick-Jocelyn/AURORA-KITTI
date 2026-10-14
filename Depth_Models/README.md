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
│   └── results/          REGISTRY.md, per-checkpoint metrics, per-weather breakdown
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
| `rank01_…best_ema_new_loss.pth` | **Headline DDCD.** AURORA TEST RMSE 1799.55 mm, DENSE 2802.79 mm |
| `rank02_…noisy_disp….pth` | Noisy-prior distillation ablation row |
| `DDCD_ablation_nonorm_v2/best_ema.pth` | Eq. 6 normalisation-isolation row |
| `CompletionFormer experiments_aurora_full/model_12.pt` | AURORA-retrained CompletionFormer |
| `BP_KITTI_Retrained/result_ema.pth` | AURORA-retrained BP-Net (RMSE_ema 1.3672) |

Note on selection: the CompletionFormer trainer saves every epoch and the pipeline always
takes the most recent, so "best" is "last" by construction — there is no validation-based
selection step.

## Known issues, carried forward deliberately

1. `benchmark/eval_batch.py` hardcodes `SEV_ORDER=["light","moderate","heavy"]`, but DENSE's
   metadata uses `"medium"` — so the DENSE Fog/medium bucket is computed and then silently
   dropped. `score_dense3.py` (in the rebuttal tooling) is the clean re-implementation that
   recovers it.
2. DENSE rain has only **two** native severity levels (15 → light, 55 → heavy). There is no
   medium rain; inventing one would not be backed by the data.
3. `benchmark/metrics_calculator_KITTI_VAL.py` is the VAL-split calculator. Its TEST
   counterpart was lost (the file on disk is 0 bytes).
