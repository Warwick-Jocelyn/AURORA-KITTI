"""
main_modality.py
================
Modality-ablation training harness for DepthCompletionNet (ResNet-34).
Trains in supervised mode (--train-mode dense) with three input options:
  --input rgbd | rgb | d

Adds DMD3C-style distillation (teacher = da3 clean depth) on top of the
standard l2 supervised loss:
  total = sup_loss + 0.1 * SSI(pred, teacher_depth) + 0.05 * grad(pred, teacher_depth)

New files created by this harness (DO NOT edit repo originals):
  distill.py               — distillation losses
  kitti_loader_modality.py — modality-aware dataloader with teacher alignment
  main_modality.py         — this file

Run from the repo root so that dataloaders/calib_cam_to_cam.txt is found.
"""

import argparse
import os
import sys
import time
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

# ---- repo imports (unchanged originals) ----------------------------------
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper

# ---- new imports ----------------------------------------------------------
from kitti_loader_modality import (
    KittiDepthModality,
    load_calib,
    oheight,
    owidth,
    input_options,
)
from distill import (
    disp_to_depth_proxy,
    ScaleShiftInvariantLoss,
    MultiScaleResidualGradientLoss,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Modality-Ablation Sparse-to-Dense + Distillation")

# ---- inherited from main.py (same defaults where applicable) -------------
parser.add_argument("-w", "--workers",      default=4,    type=int,   metavar="N")
parser.add_argument("--epochs",             default=20,   type=int,   metavar="N")
parser.add_argument("--start-epoch",        default=0,    type=int,   metavar="N")
parser.add_argument("-c", "--criterion",    default="l2", metavar="LOSS",
                    choices=criteria.loss_names)
parser.add_argument("-b", "--batch-size",   default=4,    type=int)
parser.add_argument("--lr",                 default=1e-5, type=float, metavar="LR")
parser.add_argument("--weight-decay",       default=0.0,  type=float, metavar="W")
parser.add_argument("--print-freq", "-p",   default=10,   type=int,   metavar="N")
parser.add_argument("--resume",             default="",   type=str,   metavar="PATH")
parser.add_argument("--data-folder",
                    default="/path/to/aurora-root/Dataset/AuroraKITTI/data_try",
                    type=str, metavar="PATH")
parser.add_argument("-i", "--input",        default="rgbd", type=str,
                    choices=input_options)
parser.add_argument("-l", "--layers",       default=34, type=int,
                    help="ResNet depth (18 or 34)")
parser.add_argument("--pretrained",         action="store_true",
                    help="ImageNet pre-trained weights")
parser.add_argument("--val",                default="select", type=str,
                    choices=["select", "full"])
parser.add_argument("--jitter",             default=0.1, type=float)
parser.add_argument("--rank-metric",        default="rmse", type=str,
                    choices=[m for m in dir(Result()) if not m.startswith("_")])
parser.add_argument("-m", "--train-mode",   default="dense", type=str,
                    choices=["dense"],
                    help="Only 'dense' is supported in this ablation harness")
parser.add_argument("--cpu",                action="store_true")

# ---- new ablation-specific args ------------------------------------------
parser.add_argument("--distill-ssi-weight", default=0.1, type=float,
                    help="Weight on the SSI distillation loss (default 0.1)")
parser.add_argument("--distill-grad-weight", default=0.05, type=float,
                    help="Weight on the gradient distillation loss (default 0.05)")
parser.add_argument("--no-distill",         action="store_true",
                    help="Disable distillation entirely (supervised-only baseline)")
parser.add_argument("--cap-train-samples",  default=0, type=int,
                    help="Cap train set to N samples (0=full). For smoke tests.")
parser.add_argument("--cap-val-samples",    default=0, type=int,
                    help="Cap val set to N samples (0=full). For smoke tests.")
parser.add_argument("--result",
                    default=os.path.join(
                        "/path/to/aurora-root/PhD_Project_2025/"
                        "self-supervised-depth-completion",
                        "results_modality"
                    ),
                    type=str, metavar="PATH",
                    help="Root directory for checkpoints and logs")

args = parser.parse_args()

# ---- derived flags (mirrors main.py) -------------------------------------
args.use_pose = False   # photo-metric mode is not supported here
args.use_rgb  = "rgb" in args.input
args.use_d    = "d"   in args.input
args.use_g    = False   # grayscale mode not exposed in this harness
args.w1, args.w2 = 0, 0

print(args)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"=> using '{device}' for computation.")

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

depth_criterion = (
    criteria.MaskedMSELoss() if args.criterion == "l2" else criteria.MaskedL1Loss()
)
ssi_criterion  = ScaleShiftInvariantLoss(deltas=None)
grad_criterion = MultiScaleResidualGradientLoss(deltas=None)

# ---------------------------------------------------------------------------
# WandB (graceful fallback)
# ---------------------------------------------------------------------------

_wandb_ok = False
if not os.environ.get("WANDB_DISABLED", ""):
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        wandb.init(
            project="aurora-modality-ablation",
            name=f"s2d_{args.input}_distill",
            config=vars(args),
        )
        _wandb_ok = True
        print("[wandb] initialised.")
    except Exception as e:
        print(f"[wandb] disabled: {e}")

def wandb_log(d: dict):
    if _wandb_ok:
        try:
            import wandb as _w
            _w.log(d)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_avg = AverageMeter()
    avg       = AverageMeter()
    meters    = [block_avg, avg]

    assert mode in ("train", "val")

    if mode == "train":
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    iter_count = 0
    iter_start = time.time()

    for i, batch_data in enumerate(loader):
        t0 = time.time()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        gt = batch_data.get("gt", None)
        data_time = time.time() - t0

        t1 = time.time()

        pred = model(batch_data)

        if mode == "train":
            # --- supervised loss ---
            sup_loss = depth_criterion(pred, gt)

            # --- distillation losses ---
            ssi_loss_val  = torch.tensor(0.0, device=device)
            grad_loss_val = torch.tensor(0.0, device=device)

            teacher_disp = batch_data.get("teacher_disp", None)
            if (not args.no_distill) and (teacher_disp is not None):
                gt_mask = (gt > 0)
                teacher_depth = disp_to_depth_proxy(
                    teacher_disp, gt, gt_mask
                )  # (B,1,H,W)

                ssi_losses = ssi_criterion(pred, teacher_depth)
                ssi_loss_val = sum(ssi_losses)

                grad_losses = grad_criterion(pred, teacher_depth)
                grad_loss_val = sum(grad_losses)

            total_loss = (
                sup_loss
                + args.distill_ssi_weight  * ssi_loss_val
                + args.distill_grad_weight * grad_loss_val
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- wandb per-iter ---
            wandb_log({
                "train/iter":       epoch * len(loader) + i,
                "train/sup_loss":   sup_loss.item(),
                "train/ssi_loss":   ssi_loss_val.item() if torch.is_tensor(ssi_loss_val) else float(ssi_loss_val),
                "train/grad_loss":  grad_loss_val.item() if torch.is_tensor(grad_loss_val) else float(grad_loss_val),
                "train/total_loss": total_loss.item(),
            })

        gpu_time = time.time() - t1
        iter_count += 1

        with torch.no_grad():
            mini_bs = next(iter(batch_data.values())).size(0)
            result  = Result()
            if gt is not None:
                result.evaluate(pred.data, gt.data, 0)
            [m.update(result, gpu_time, data_time, mini_bs) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_avg, avg)

    # --- per-epoch val log ---
    if mode == "val":
        epoch_avg = avg.average()
        wandb_log({"val/rmse": epoch_avg.rmse, "epoch": epoch})

    # --- it/s report every epoch ---
    elapsed = time.time() - iter_start
    its = iter_count / elapsed if elapsed > 0 else 0.0
    print(
        f"[{mode.upper()} ep{epoch}] {iter_count} iters in {elapsed:.1f}s "
        f"=> {its:.2f} it/s  (batch={args.batch_size})"
    )

    avg_result = logger.conditional_save_info(mode, avg, epoch)
    is_best    = logger.rank_conditional_save_best(mode, avg_result, epoch)
    if is_best and mode == "val":
        # guard: img_merge may not be set when val set is small (< 800 samples)
        if hasattr(logger, "img_merge"):
            logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg_result, is_best)
    return avg_result, is_best


# ---------------------------------------------------------------------------
# Teacher-alignment sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_alignment_check(loader, max_batches: int = 5):
    """
    After augmentation, verify that the affine-fitted teacher_depth correlates
    with GT at valid pixels.  Reports Pearson r and median absolute relative
    error (MedAbsRel) averaged over the checked batches.
    """
    print("\n[SANITY] Teacher-alignment check ...")
    all_r    = []
    all_mare = []
    checked  = 0
    eps      = 1e-6

    for batch_data in loader:
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        gt   = batch_data.get("gt")
        disp = batch_data.get("teacher_disp")
        if gt is None or disp is None:
            break

        mask = (gt > 0)                         # (B,1,H,W) bool
        teacher_depth = disp_to_depth_proxy(disp, gt, mask)

        # flatten valid pixels across batch
        gt_v   = gt[mask].cpu().numpy()
        td_v   = teacher_depth[mask].cpu().numpy()

        # Pearson correlation
        if len(gt_v) > 10:
            r = float(np.corrcoef(gt_v, td_v)[0, 1])
            all_r.append(r)

        # Median AbsRel
        mare = float(np.median(np.abs(td_v - gt_v) / (gt_v + eps)))
        all_mare.append(mare)

        checked += 1
        if checked >= max_batches:
            break

    if all_r:
        print(f"[SANITY]  Pearson r (teacher_depth vs gt)  : {np.mean(all_r):.4f}  (over {checked} batches)")
    else:
        print("[SANITY]  Could not compute Pearson r (too few valid pixels)")
    if all_mare:
        print(f"[SANITY]  Median AbsRel (teacher_depth vs gt): {np.mean(all_mare):.4f}  (<0.3 expected)")
    print("[SANITY] Done.\n")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    global args

    # ---- resume / evaluate ------------------------------------------------
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}' ...")
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch   = checkpoint["epoch"] + 1
            print(f"   resumed from epoch {checkpoint['epoch']}.")
        else:
            print(f"No checkpoint at '{args.resume}'")
            return

    # ---- model + optimizer -----------------------------------------------
    print("=> creating model and optimizer ...")
    model = DepthCompletionNet(args).to(device)
    model_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer    = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("   checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    # ---- data loaders ----------------------------------------------------
    print("=> creating data loaders ...")
    train_dataset = KittiDepthModality("train", args, cap_samples=args.cap_train_samples)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=cuda,
    )
    print(f"\t==> train_loader size: {len(train_loader)}")

    val_dataset = KittiDepthModality("val", args, cap_samples=args.cap_val_samples)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=cuda,
    )
    print(f"\t==> val_loader size: {len(val_loader)}")

    # ---- logger ----------------------------------------------------------
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint["best_result"]
    print("=> logger created.")

    # ---- teacher alignment sanity (train split, a few batches) ----------
    if not args.no_distill:
        teacher_alignment_check(train_loader, max_batches=5)

    # ---- main training loop ----------------------------------------------
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"=> starting training epoch {epoch} ..")
        iterate("train", args, train_loader, model, optimizer, logger, epoch)
        result, is_best = iterate("val",   args, val_loader,   model, None,      logger, epoch)
        # qualitative weather-VAL visualization to wandb (clear/fog/rain/snow panels)
        if _wandb_ok:
            try:
                from wandb_val_viz import log_val_viz
                _m = model.module if hasattr(model, "module") else model
                _n = log_val_viz(_m, args, epoch, device, wandb)
                print(f"[wandb viz] logged {_n} val panels for epoch {epoch}")
            except Exception as _e:
                print(f"[wandb viz] skip: {_e}")
        helper.save_checkpoint(
            {
                "epoch":       epoch,
                "model":       model.module.state_dict(),
                "best_result": logger.best_result,
                "optimizer":   optimizer.state_dict(),
                "args":        args,
            },
            is_best,
            epoch,
            logger.output_directory,
        )

    if _wandb_ok:
        try:
            import wandb as _w
            _w.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
