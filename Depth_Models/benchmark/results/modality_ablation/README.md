# Modality Ablation (RC 1.4) — results layout

Canonical save locations (point all runs/eval here):

- Training checkpoints + logs (per variant): repo `self-supervised-depth-completion/results_modality/mode=dense.input=<rgbd|rgb|d>.resnet34.../`
- wandb: project `aurora-modality-ablation`, runs `s2d_<input>_distill` (online)
- VAL predictions (16-bit PNG, pred*256): `preds_<input>/`
- Metrics per variant (standard + denoising ERR/DCE/SR/NRR, per weather): `metrics/denoise_<input>.{csv,md}`
- Combined comparison (Table B, 3 rows): `metrics/Table_B_modality.md`
- Smoke outputs (throwaway): `_smoke/`

Eval command (per variant), run after training:
  python eval_denoise.py --set VAL --pred_dir <preds_X> --pred_scale 256 --out_prefix metrics/denoise_<X>
