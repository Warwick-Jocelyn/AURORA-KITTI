# -*- coding: utf-8 -*-
# train_wandb.py (energy-weighted distillation version)

import os
from datetime import datetime
import math

import torch
from tqdm import tqdm
import hydra
import torch.distributed as dist
import numpy as np
import wandb
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

from utils import *  # Trainer, AverageMeter, etc.

from dmd3c_distill_loss import (
    disp_to_depth_proxy,
    ScaleShiftInvariantLoss,
    MultiScaleResidualGradientLoss,
    compute_energy_weight_map,
)


def get_max_images(cfg, key):
    val = OmegaConf.select(cfg, key, default=None)
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ["full", "all", "none", ""]:
            return None
        try:
            v_int = int(v)
            return v_int if v_int > 0 else None
        except ValueError:
            return None
    if isinstance(val, (int, float)):
        return int(val) if val > 0 else None
    return None


def colorize_depth(depth_m, vmin=0.0, vmax=80.0, cmap='turbo'):
    depth_m = np.squeeze(depth_m)
    norm = (depth_m - vmin) / (vmax - vmin + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    rgb = (cm(norm)[..., :3] * 255).astype(np.uint8)
    rgb[depth_m <= 0] = 0
    return rgb


def colorize_weight(w_map, cmap='viridis'):
    """
    w_map: 2D numpy [0,1]
    """
    w_map = np.squeeze(w_map)
    w_map = np.clip(w_map, 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    rgb = (cm(w_map)[..., :3] * 255).astype(np.uint8)
    return rgb


def train(run):
    deltas = (2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2),
              2 ** (-2 * 2), 2 ** (-1 * 2), 1)

    compute_ssi_loss = ScaleShiftInvariantLoss(deltas=deltas)
    compute_reg_loss = MultiScaleResidualGradientLoss(deltas=deltas, K=4)

    lambda_distill = getattr(run.cfg, "lambda_distill", 0.1)
    lambda_grad = getattr(run.cfg, "lambda_grad", 0.05)

    # energy hyperparams (safe defaults)
    energy_patch = getattr(run.cfg, "energy_patch", 16)
    energy_beta = getattr(run.cfg, "energy_beta", 1.0)
    energy_lambda_s = getattr(run.cfg, "energy_lambda_s", 1.0)
    energy_lambda_a = getattr(run.cfg, "energy_lambda_a", 1.0)
    energy_w_min = getattr(run.cfg, "energy_w_min", 0.3)
    energy_kmin = getattr(run.cfg, "energy_kmin", 20)
    energy_tau_mode = getattr(run.cfg, "energy_tau_mode", "median")

    max_train_imgs = get_max_images(run.cfg, "max_train_images")
    num_train_imgs = 0

    sup_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()
    grad_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    printed_data_info = False

    pbar = tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank)

    for datas in pbar:
        if (max_train_imgs is not None) and (num_train_imgs >= max_train_imgs):
            break

        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)

        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)

        # ----------------- unpack (train may return 6-tuple) -----------------
        if len(datas) == 6:
            rgb, disp_clean, disp_noisy, lidar, K_cam, gt_depth = datas
        else:
            rgb, disp_clean, lidar, K_cam, gt_depth = datas
            disp_noisy = disp_clean  # fallback

        if (not printed_data_info) and run.rank == 0:
            print("\n[Train Data Check]")
            print(f"  RGB:        shape={rgb.shape}, range=[{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"  disp_clean: shape={disp_clean.shape}, range=[{disp_clean.min():.3f}, {disp_clean.max():.3f}]")
            print(f"  disp_noisy: shape={disp_noisy.shape}, range=[{disp_noisy.min():.3f}, {disp_noisy.max():.3f}]")
            print(f"  LiDAR:      shape={lidar.shape}, range=[{lidar.min():.3f}, {lidar.max():.3f}]")
            print(f"  K_cam:      shape={K_cam.shape}")
            print(f"  GT depth:   shape={gt_depth.shape}, range=[{gt_depth.min():.3f}, {gt_depth.max():.3f}]\n")
            if max_train_imgs is not None:
                print(f"[Train] Using at most {max_train_imgs} images per rank this epoch.")
            printed_data_info = True

        # ----------------- Forward (keep BPNet interface clean) -----------------

        # output = net(rgb, disp_clean, lidar, K_cam)
        output = run.net(rgb, disp_clean, lidar, K_cam)

        # ----------------- Supervised loss (sparse GT) -----------------
        sup_loss = run.criterion(output, gt_depth)

        # ----------------- teacher & energy weight map -----------------
        gt_mask = (gt_depth > 0)

        with torch.no_grad():
            teacher_clean = disp_to_depth_proxy(disp_clean, gt_depth, gt_mask, eps=1e-6, min_points=50)
            teacher_noisy = disp_to_depth_proxy(disp_noisy, gt_depth, gt_mask, eps=1e-6, min_points=50)

            w_map, w_dbg = compute_energy_weight_map(
                teacher_clean=teacher_clean,
                teacher_noisy=teacher_noisy,
                gt_depth=gt_depth,
                gt_mask=gt_mask,
                patch=energy_patch,
                beta=energy_beta,
                lambda_s=energy_lambda_s,
                lambda_a=energy_lambda_a,
                w_min=energy_w_min,
                k_min_points=energy_kmin,
                tau_mode=energy_tau_mode,
            )

        # distill toward clean teacher, weighted by energy reliability
        distill_loss = compute_ssi_loss(output, teacher_clean, valid_mask=None, weight_map=w_map)
        grad_loss = compute_reg_loss(output, teacher_clean, valid_mask=None, weight_map=w_map)

        # ----------------- logging scalars -----------------
        if isinstance(sup_loss, (list, tuple)):
            sup_total = sum(l.detach().item() for l in sup_loss)
        else:
            sup_total = sup_loss.detach().item()

        distill_total = sum(l.detach().item() for l in distill_loss) if isinstance(distill_loss, (list, tuple)) else distill_loss.detach().item()
        grad_total = sum(l.detach().item() for l in grad_loss) if isinstance(grad_loss, (list, tuple)) else grad_loss.detach().item()

        total_for_log = sup_total + lambda_distill * distill_total + lambda_grad * grad_total

        sup_loss_meter.update(sup_total)
        distill_loss_meter.update(distill_total)
        grad_loss_meter.update(grad_total)
        total_loss_meter.update(total_for_log)

        # ----------------- Backward (keep your multi-scale style) -----------------
        if isinstance(sup_loss, (list, tuple)):
            loss_list = []
            for i, ls in enumerate(sup_loss):
                ld = distill_loss[i] if isinstance(distill_loss, (list, tuple)) else distill_loss
                lg = grad_loss[i]    if isinstance(grad_loss, (list, tuple))    else grad_loss
                loss_list.append(ls + lambda_distill * ld + lambda_grad * lg)
            sum(loss_list).backward()
            loss = loss_list
        else:
            d_scalar = sum(distill_loss) if isinstance(distill_loss, (list, tuple)) else distill_loss
            g_scalar = sum(grad_loss)    if isinstance(grad_loss, (list, tuple))    else grad_loss
            loss = sup_loss + lambda_distill * d_scalar + lambda_grad * g_scalar
            loss.backward()

        if run.clip:
            grad_norm = run.clip(run.net.parameters())
        else:
            grad_norm = None

        run.optimizer.step()
        run.net_ema.update(run.net)

        if run.lr_iter:
            run.lr_scheduler.step()

        # ----------------- TensorBoard -----------------
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            if isinstance(loss, (list, tuple)):
                run.writer.add_scalars("Loss", {f"{idx}": l.item() for idx, l in enumerate(loss)}, run.iter)
            else:
                run.writer.add_scalars("Loss", {"0": loss.item()}, run.iter)

            if run.clip and (grad_norm is not None):
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)

        # ----------------- wandb logging + weight map visualization -----------------
        if (run.rank == 0) and (wandb.run is not None) and (run.iter % run.cfg.vis_iter == 0):
            log_dict = {
                "train/lr": run.optimizer.param_groups[0]['lr'],
                "train/loss_supervised": sup_total,
                "train/loss_ssi_raw": distill_total,
                "train/loss_reg_raw": grad_total,
                "train/loss_total": total_for_log,
                "train/w_mean": float(w_map.mean().item()),
                "train/w_min": float(w_map.min().item()),
                "train/w_max": float(w_map.max().item()),
                "train/tau_s": float(w_dbg["tau_s"].item()) if hasattr(w_dbg["tau_s"], "item") else float(w_dbg["tau_s"]),
                "train/tau_a": float(w_dbg["tau_a"].item()) if hasattr(w_dbg["tau_a"], "item") else float(w_dbg["tau_a"]),
                "train/iter": run.iter,
                "epoch": run.epoch,
            }

            # visualize first sample weight map
            w0 = w_map[0, 0].detach().cpu().numpy()
            vis_w = colorize_weight(w0, cmap='viridis')
            log_dict["train/weight_map"] = wandb.Image(vis_w, caption="energy-weighted reliability w(x)")

            wandb.log(log_dict)

        num_train_imgs += rgb.shape[0]
        run.iter += 1

        pbar.set_postfix(
            Sup=f"{sup_total:.3f}",
            SSI=f"{distill_total:.3f}",
            Reg=f"{grad_total:.3f}",
            Total=f"{total_for_log:.3f}",
            N=num_train_imgs,
        )

    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()

    if run.rank == 0:
        print(f"[Train Epoch {run.epoch}] "
              f"Sup: {sup_loss_meter.avg:.4f}, "
              f"SSI(raw): {distill_loss_meter.avg:.4f}, "
              f"Reg(raw): {grad_loss_meter.avg:.4f}, "
              f"Total(lambda_d={lambda_distill}, lambda_g={lambda_grad}): {total_loss_meter.avg:.4f}")
        if wandb.run is not None:
            wandb.log({
                "train_epoch/loss_supervised": sup_loss_meter.avg,
                "train_epoch/loss_ssi_raw": distill_loss_meter.avg,
                "train_epoch/loss_reg_raw": grad_loss_meter.avg,
                "train_epoch/loss_total": total_loss_meter.avg,
                "epoch": run.epoch,
            })


def test(run, iter=False):
    net = run.net_ema.module
    best_metric_name = "best_metric_ema"
    net.eval()

    metric_meters = None
    metric_names = getattr(run.metric, "metric_name", None)
    vis_done = False

    max_val_imgs = get_max_images(run.cfg, "max_val_images")
    num_val_imgs = 0
    vis_samples = []

    with torch.no_grad():
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            if (max_val_imgs is not None) and (num_val_imgs >= max_val_imgs):
                break

            datas = run.init_cuda(*datas)

            # val/test expected 5-tuple: (rgb, disp_clean, lidar, K, depth)
            rgb, disp_clean, lidar, K_cam, gt_depth = datas

            output = net(rgb, disp_clean, lidar, K_cam)

            output_vis = output[-1] if isinstance(output, (list, tuple)) else output

            metrics_out = run.metric(output_vis, gt_depth)
            if not isinstance(metrics_out, (list, tuple)):
                metrics_out = [metrics_out]

            if metric_meters is None:
                metric_meters = [AverageMeter() for _ in range(len(metrics_out))]
                if (metric_names is None) or (len(metric_names) != len(metrics_out)):
                    metric_names = [f"metric_{i}" for i in range(len(metrics_out))]

            for i, m in enumerate(metrics_out):
                v = m.clone().float() if isinstance(m, torch.Tensor) else torch.tensor(m, device=output_vis.device, dtype=torch.float32)
                if v.dim() > 0:
                    v = v.mean()
                if run.ddp:
                    dist.reduce(v, 0, dist.ReduceOp.AVG)
                metric_meters[i].update(v.item())

            num_val_imgs += rgb.shape[0]

            # visualization
            if (not iter) and (run.rank == 0) and (not vis_done):
                gt_mask = (gt_depth > 0)
                teacher_depth = disp_to_depth_proxy(disp_clean, gt_depth, gt_mask, eps=1e-6, min_points=50)

                B = rgb.shape[0]
                for b in range(B):
                    vis_samples.append({
                        "rgb": rgb[b].detach().cpu(),
                        "lidar": lidar[b].detach().cpu(),
                        "gt_depth": gt_depth[b].detach().cpu(),
                        "pred": output_vis[b].detach().cpu(),
                        "teacher": teacher_depth[b].detach().cpu(),
                    })

                if len(vis_samples) >= 8:
                    n_examples = 8
                    fig, axs = plt.subplots(n_examples, 5, figsize=(24, 2.5 * n_examples))
                    if n_examples == 1:
                        axs = np.expand_dims(axs, axis=0)

                    col_titles = ["RGB", "LiDAR", "Teacher(depth-proxy)", "GT Depth", "Pred"]

                    for row in range(n_examples):
                        sample = vis_samples[row]
                        rgb_i = sample["rgb"].permute(1, 2, 0).numpy()
                        lidar_i = sample["lidar"][0].numpy()
                        gt_i = sample["gt_depth"][0].numpy()
                        pred_i = sample["pred"][0].numpy()
                        teacher_i = sample["teacher"][0].numpy()

                        rgb_i = (rgb_i - rgb_i.min()) / (rgb_i.max() - rgb_i.min() + 1e-6)

                        vis_lidar = colorize_depth(lidar_i, vmin=0, vmax=80, cmap='turbo')
                        vis_teacher = colorize_depth(teacher_i, vmin=0, vmax=80, cmap='turbo')
                        vis_gt = colorize_depth(gt_i, vmin=0, vmax=80, cmap='turbo')
                        vis_pred = colorize_depth(
                            pred_i,
                            vmin=np.percentile(pred_i, 5),
                            vmax=np.percentile(pred_i, 95),
                            cmap='turbo'
                        )

                        axs[row, 0].imshow(rgb_i)
                        axs[row, 1].imshow(vis_lidar)
                        axs[row, 2].imshow(vis_teacher)
                        axs[row, 3].imshow(vis_gt)
                        axs[row, 4].imshow(vis_pred)
                        for col in range(5):
                            axs[row, col].axis("off")

                    for j, t in enumerate(col_titles):
                        axs[0, j].set_title(t)

                    plt.tight_layout()
                    if wandb.run is not None:
                        wandb.log({"val/vis_8examples_5panels": wandb.Image(fig)}, commit=False)
                    plt.close(fig)
                    vis_done = True

    if metric_meters is None:
        return

    legand = 'net_ema'
    top1_avg = metric_meters[0].avg
    best_metric_value = getattr(run, best_metric_name, float('inf'))

    if iter:
        for name, meter in zip(metric_names, metric_meters):
            run.writer.add_scalars(f"{name}_Iter", {legand: meter.avg}, run.iter)
    else:
        for name, meter in zip(metric_names, metric_meters):
            run.writer.add_scalars(f"{name}", {legand: meter.avg}, run.epoch)

    if not iter:
        if run.rank == 0:
            run.ddp_cout(f"[VAL] Epoch {run.epoch} primary metric={top1_avg} (prev best={best_metric_value})")

        if math.isfinite(top1_avg):
            is_new_best = (not math.isfinite(best_metric_value)) or (top1_avg < best_metric_value)
            if is_new_best:
                setattr(run, best_metric_name, top1_avg)
                run.save_state()

                if run.rank == 0:
                    best_metrics = {name: meter.avg for name, meter in zip(metric_names, metric_meters)}
                    run.best_metrics_all = best_metrics
                    run.best_epoch = run.epoch

                    base_dir = get_original_cwd()
                    ckpt_dir = os.path.join(base_dir, 'checkpoint', run.cfg.name)
                    os.makedirs(ckpt_dir, exist_ok=True)

                    ckpt_path = os.path.join(ckpt_dir, 'best_ema.pth')
                    run.best_ckpt_path = ckpt_path

                    run.ddp_cout(f"Epoch {run.epoch} NEW BEST {best_metric_name}: {top1_avg:.7f}\n"
                                 f"  -> EMA checkpoint saved to: {ckpt_path}")

                    metrics_txt = os.path.join(ckpt_dir, 'best_metrics.txt')
                    with open(metrics_txt, 'w') as f:
                        f.write(f"Best epoch: {run.epoch}\n")
                        f.write(f"Best primary metric (metric_0 / RMSE): {top1_avg:.7f}\n")
                        f.write("MetricsAll:\n")
                        for name, val in best_metrics.items():
                            f.write(f"  {name}: {val:.7f}\n")
                        f.write(f"Checkpoint path: {ckpt_path}\n")
                    run.best_metrics_file = metrics_txt

                    if wandb.run is not None:
                        log_dict = {"best/epoch": run.epoch, "best/metric0": top1_avg, "best/ckpt_path": ckpt_path}
                        for name, val in best_metrics.items():
                            log_dict[f"best/{name}"] = val
                        wandb.log(log_dict, commit=False)

    if (run.rank == 0) and (wandb.run is not None):
        log_dict = {"iter": run.iter, "epoch": run.epoch} if iter else {"epoch": run.epoch}
        for name, meter in zip(metric_names, metric_meters):
            log_dict[(f"val_iter/{name}" if iter else f"val/{name}")] = meter.avg
        wandb.log(log_dict)


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        enable_wandb = OmegaConf.select(cfg, "use_wandb", default=True)

        if (run.rank == 0) and enable_wandb:
            run_name = f"{cfg.get('name', 'BPNet')}_{datetime.now().strftime('%m%d_%H%M')}"
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(project="kitti-dmd3c-bpnet", name=run_name, config=cfg_dict)

            try:
                n_train = len(run.trainloader.dataset)
                n_val = len(run.testloader.dataset)
            except Exception:
                n_train = len(run.trainloader)
                n_val = len(run.testloader)
            print(f"[DATA] num_train = {n_train}, num_val = {n_val}")

        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)

            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()

            epoch_idx = epoch + 1
            if epoch_idx % 2 == 0:
                filename = f"epoch_{epoch_idx:03d}_ema.pth"
                run.save_state(filename=filename)
                if run.rank == 0:
                    base_dir = get_original_cwd()
                    ckpt_dir = os.path.join(base_dir, 'checkpoint', run.cfg.name)
                    run.ddp_cout(f"[Epoch {epoch_idx}] Periodic checkpoint saved to: {os.path.join(ckpt_dir, filename)}")

        if run.rank == 0:
            if hasattr(run, "best_metrics_all"):
                print("\n========== Training Finished ==========")
                print(f"Best epoch: {getattr(run, 'best_epoch', 'N/A')}")
                print(f"Best primary metric (metric_0 / RMSE): {run.best_metric_ema:.7f}")
                print("MetricsAll at best checkpoint:")
                for name, val in run.best_metrics_all.items():
                    print(f"  {name}: {val:.7f}")
                if hasattr(run, "best_ckpt_path"):
                    print(f"Best EMA checkpoint path: {run.best_ckpt_path}")
                if hasattr(run, "best_metrics_file"):
                    print(f"Best metrics saved to: {run.best_metrics_file}")
            else:
                print("\n[WARN] No validation metrics were computed; best checkpoint was never updated.")

            if wandb.run is not None and hasattr(run, "best_metrics_all"):
                wandb.run.summary["best/epoch"] = getattr(run, "best_epoch", -1)
                wandb.run.summary["best/primary_metric"] = run.best_metric_ema
                if hasattr(run, "best_ckpt_path"):
                    wandb.run.summary["best/ckpt_path"] = run.best_ckpt_path
                for name, val in run.best_metrics_all.items():
                    wandb.run.summary[f"best/{name}"] = val

        if run.rank == 0 and wandb.run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()
