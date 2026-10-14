# Upstream repositories and pinned commits

Each baseline below is a public repository. We do **not** redistribute them. To reproduce our
environment: clone at the pinned commit, then copy the matching folder from `baselines/` over
the clone.

All commits were recorded from the working trees on 2026-10-13/14.

| Baseline | Upstream | Pinned commit | Our changes |
|---|---|---|---|
| BP-Net | `github.com/kakaxi314/BP-Net` | — (upstream HEAD, no local commits) | `train.py`, `test.py`, `datasets/kitti.py`, `models/__init__.py`, configs |
| DMD3C | CVPR'25 "Distilling Monocular Foundation Model for Fine-grained Depth Completion" | see `DDCD/` | the whole DDCD method |
| CompletionFormer | `github.com/youmi-zym/CompletionFormer` | see report | AURORA training + chain scripts |
| NLSPN | `github.com/zzangjinsun/NLSPN_ECCV20` | — | eval adaptation only; no training |
| Sparse2Dense | `github.com/fangchangma/self-supervised-depth-completion` | upstream HEAD, no local commits | `main_modality.py`, `kitti_loader_modality.py`, `distill.py`, inference + metrics |
| OMNI-DC | `github.com/princeton-vl/OMNI-DC` | `955c113aaf4007e22e595182b2b07c285fecf3d3` | `run_kitti_test_inproc.py` (6–9 h → 20 min; fixes ×1000 PNG saturation at 65.5 m), 45-line patch to `config.py`/`demo.py` |
| Marigold-DC | `github.com/prs-eth/Marigold-DC` | `ffca15c8585dab418425701207f136f009dee545` | `marigold_dc_batch.py`, `evaluate_kitti_depth16.py`, colourisation |
| PromptDA | `github.com/DepthAnything/PromptDA` | `b05523cf9c27e2dbce578b9003a8906bf8ed2eb3` | `demo_promptda_kitti.py` only — **zero tracked modifications** |
| UniK3D | `github.com/lpiccinelli-eth/UniK3D` | `29f7862` | 16-bit ×256 depth saver in `scripts/demo.py`, inference + benchmark drivers |
| Depth-Anything-V2 | `github.com/DepthAnything/Depth-Anything-V2` | `31dc9770` | batch runner + 16-bit postprocess |
| Depth-Anything-3 | `github.com/ByteDance-Seed/Depth-Anything-3` | `2c21ea84` | **zero tracked diffs**; our batch/eval drivers only |
| IP-Basic | `github.com/kujason/ip_basic` | `b6e6ce9c64e1fee63c2535f9999503cfe7e68066` | 2-line NumPy-2 fix (`np.bool` → `bool`, `depth_map_utils.py:208,225`) + AURORA eval harness |
| rain-rendering *(simulation)* | `github.com/astra-vision/rain-rendering` | `0a9d8ca739d71d4b09f61567c6f6cea94a79868e`, submodule `weather-particle-simulator@19fa03b419a32902abbf0ec899ff0ca6e77b56fa` | debug prints only. Note `.gitmodules` points at the dead `cv-rits` org; the live remote is `astra-vision` |

## Pretrained weights (re-downloadable, not redistributed)

| Model | Source |
|---|---|
| Depth-Anything-V2 ViT-L / ViT-S | HF `depth-anything/Depth-Anything-V2-Large` / `-Small` |
| Depth-Anything-3 | HF `ByteDance-Seed/Depth-Anything-3` |
| OMNI-DC v1.0 / v1.1 | HF `zuoym15/OMNI-DC` (or the Drive links in its README) |
| PromptDA ViT-L | HF `depth-anything/prompt-depth-anything-vitl` |
| UniK3D ViT-L | `from_pretrained("lpiccinelli/unik3d-vitl")` at runtime — nothing stored locally |
| BP-Net KITTI | upstream release tarball |
| NLSPN KITTI-DC | upstream release |

## Datasets

| Dataset | Source |
|---|---|
| KITTI depth completion | `data_depth_{annotated,velodyne,selection}.zip` from the KITTI S3 bucket; raw RGB via the repo's own `rgb_{train,val}_downloader.sh` (156 drives) |
| DENSE / Seeing Through Fog | `uni-ulm.de` / the STF project page (registration required) |
| MUSES | `muses.vision.ee.ethz.ch` (gated) |
| AURORA-KITTI | see the repository root README |
