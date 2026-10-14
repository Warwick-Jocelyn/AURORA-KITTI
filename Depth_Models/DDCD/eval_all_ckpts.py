"""
Evaluate every unique checkpoint listed in CKPT_LIST on the
val_selection_cropped_DENSE dataset and emit one JSON line per checkpoint.

Usage:
    cd /path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C
    export PYTHONPATH=$(realpath ..):$PYTHONPATH
    python eval_all_ckpts.py
"""
import os
import sys
import json
import time
import inspect
import torch
from tqdm import tqdm
from PIL import Image
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch.distributed as dist

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_BP_NET_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _BP_NET_DIR not in sys.path:
    sys.path.append(_BP_NET_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import Trainer, AverageMeter
from test import _pick_inputs_for_forward

CKPT_LIST = [
    # (md5, label, abs_path)
    ("b4a59bac",
     "pretrained_mixed_singleview_256",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/pretrained_mixed_singleview_256.pth"),
    ("596b83b1",
     "dmd3c_distillation_depth_anything_v2",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/BP_KITTI_noisy_disp/dmd3c_distillation_depth_anything_v2.pth"),
    ("12c7417d",
     "BP_KITTI_result_ema",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/checkpoints/BP_KITTI/result_ema.pth"),
    ("5d066495",
     "DMD3D_BP_KITTI_best_ema_V1",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/DMD3D_BP_KITTI/best_ema_V1.pth"),
    ("64d4991d",
     "DMD3D_BP_KITTI_best_ema_V2",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/DMD3D_BP_KITTI/best_ema_V2.pth"),
    ("b15b8094",
     "DMD3D_BP_KITTI_best_ema_V3",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/DMD3D_BP_KITTI/best_ema_V3.pth"),
    ("5ede5fab",
     "DMD3D_BP_KITTI_best_ema_for_val",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/DMD3D_BP_KITTI/best_ema_for_val.pth"),
    ("035f1ab4",
     "DMD3D_BP_KITTI_best_ema_new_loss",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoints/DMD3D_BP_KITTI/best_ema_new_loss.pth"),
    ("d574f2b4",
     "DMD3D_BP_KITTI_noisy_disp_best_ema_noisy_dis",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/DMD3D_BP_KITTI_noisy_disp/best_ema_noisy_dis.pth"),
    ("6bd8e5bc",
     "DMD3D_BP_KITTI_Aurora_full_energy_best_ema",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/DMD3D_BP_KITTI_Aurora_full_energy/best_ema.pth"),
    ("64ee5ead",
     "DMD3D_Aurora_DA3_best_ema_3954_in_DENSE",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/DMD3D_Aurora_DA3_pretrained/best_ema_3954_in_DENSE.pth"),
    ("f04b6e4f",
     "DMD3D_Aurora_DA3_best_ema_V2",
     "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/checkpoint/DMD3D_Aurora_DA3_pretrained/best_ema_V2.pth"),
]

OUT_PATH = os.path.join(_THIS_DIR, "ckpt_eval_results.jsonl")


def evaluate_once(run):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()

    metric_names = list(getattr(run.metric, "metric_name", ["RMSE"]))
    meters = [AverageMeter() for _ in metric_names]
    n = 0
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader, desc="eval", leave=False, dynamic_ncols=True)):
            datas = run.init_cuda(*datas)
            inputs = _pick_inputs_for_forward(datas, net, run=None)
            output = net(*inputs)
            pred = output[-1] if isinstance(output, (list, tuple)) else output
            metrics_out = run.metric(pred, datas[-1])
            if not isinstance(metrics_out, (list, tuple)):
                metrics_out = [metrics_out]
            for i, m in enumerate(metrics_out):
                v = m.clone().float() if isinstance(m, torch.Tensor) else torch.tensor(m, device=pred.device, dtype=torch.float32)
                if v.dim() > 0:
                    v = v.mean()
                meters[i].update(v.item())
            n += pred.shape[0]
    return {name: meter.avg for name, meter in zip(metric_names, meters)}, n


def build_cfg():
    # Load same configs as Hydra would, with overrides for selval / MetricALL.
    with initialize(version_base="1.2", config_path="configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "metric=MetricALL",
                "data=KITTI",
                "data.testset.mode=selval",
                "test_batch_size=2",
                "num_workers=1",
                "name=CKPT_EVAL",
                "gpus=[0]",
                # we don't need real training data for this; but Trainer still
                # instantiates trainset. Keep it small by NOT touching it -
                # the trainset glob just enumerates filenames so it's fine.
            ],
        )
    return cfg


def main():
    out_f = open(OUT_PATH, "w")
    cfg = build_cfg()
    OmegaConf.set_struct(cfg, False)

    for md5, label, ckpt_path in CKPT_LIST:
        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] missing: {ckpt_path}")
            continue
        print(f"\n=== Evaluating [{label}]  md5={md5}\n    {ckpt_path}")
        cfg.chpt = ckpt_path
        cfg.name = f"EVAL_{md5}"
        t0 = time.time()
        try:
            with Trainer(cfg) as run:
                metrics, n = evaluate_once(run)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            row = {"md5": md5, "label": label, "ckpt": ckpt_path,
                   "error": repr(e), "elapsed_s": time.time() - t0}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            continue
        elapsed = time.time() - t0
        row = {"md5": md5, "label": label, "ckpt": ckpt_path,
               "n_samples": n, "elapsed_s": round(elapsed, 1), **metrics}
        print(f"[OK] {label}: RMSE={metrics.get('RMSE'):.4f}  "
              f"MAE={metrics.get('MAE'):.4f}  iRMSE={metrics.get('iRMSE'):.4f}  "
              f"iMAE={metrics.get('iMAE'):.4f}  ({elapsed:.0f}s, n={n})")
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        torch.cuda.empty_cache()

    out_f.close()
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
