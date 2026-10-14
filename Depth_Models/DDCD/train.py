# -*- coding: utf-8 -*-
# @File   : train_wandb.py
# @Project: BP-Net (+ DMD3C distillation, W&B logging)
# @Desc   : 正式训练版本：支持控制 train/val 使用样本数（或 full），
#           GT 监督 + 蒸馏 (RandomProposalNormalizationLoss)，
#           保存 EMA 上 MetricALL 主指标最好的 checkpoint，并在本地 & wandb 记录 MetricsAll。

import os
import sys
from datetime import datetime

# 确保 BP-Net 目录在 sys.path 中，使 models 包可被 Hydra 找到
_bp_net_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _bp_net_dir not in sys.path:
    sys.path.insert(0, _bp_net_dir)

import torch
from tqdm import tqdm
import hydra
import torch.distributed as dist
import numpy as np
import wandb
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from utils import *          # Trainer, AverageMeter, etc.
# from rpnloss import RandomProposalNormalizationLoss

from hydra.utils import get_original_cwd
import math
# from gradloss import GradientMatchingLoss

from dmd3c_distill_loss import disp_to_depth_proxy, ScaleShiftInvariantLoss, MultiScaleResidualGradientLoss

# ---------------------------------------------------------------------
# 小工具：从 cfg 中解析 max_*_images（支持 int / "full" / 未设置）
# ---------------------------------------------------------------------
def get_max_images(cfg, key):
    """
    从 cfg 中解析最大使用的图像数量：
      - 如果是 int 且 > 0，则返回该值
      - 如果是 "full" / "all" / "none" / <=0 / 未设置，则返回 None（表示不限制）
    """
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

# ---------------------------------------------------------------------
# 可视化：深度 -> 伪彩色
# ---------------------------------------------------------------------
def colorize_depth(depth_m, vmin=0.0, vmax=80.0, cmap='turbo'):
    """
    depth_m: 2D numpy array, 单位可以是 meter / relative depth 等
    vmin, vmax: 映射到 colormap 的范围
    """
    depth_m = np.squeeze(depth_m)
    norm = (depth_m - vmin) / (vmax - vmin + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    rgb = (cm(norm)[..., :3] * 255).astype(np.uint8)
    rgb[depth_m <= 0] = 0
    return rgb

def train(run):
    # 多尺度权重（你的网络如果是 6 个尺度输出，默认这个长度就匹配）
    deltas = (2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2),
              2 ** (-2 * 2), 2 ** (-1 * 2), 1)

    compute_ssi_loss = ScaleShiftInvariantLoss(deltas=deltas)
    compute_reg_loss = MultiScaleResidualGradientLoss(deltas=deltas, K=4)

    # 蒸馏 loss 权重（命令行 ++lambda_distill=0.1）
    lambda_distill = getattr(run.cfg, "lambda_distill", 0.1)
    # 梯度正则权重（命令行 ++lambda_grad=0.05）
    lambda_grad = getattr(run.cfg, "lambda_grad", 0.05)

    # train 使用样本数控制：如果为 None 表示 full
    max_train_imgs = get_max_images(run.cfg, "max_train_images")
    num_train_imgs = 0

    sup_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()       # 未加权 SSI
    grad_loss_meter = AverageMeter()          # 未加权 reg
    total_loss_meter = AverageMeter()         # 实际反传 total

    printed_data_info = False

    pbar = tqdm(run.trainloader,
                desc="train",
                dynamic_ncols=True,
                leave=False,
                disable=run.rank)

    for datas in pbar:
        if (max_train_imgs is not None) and (num_train_imgs >= max_train_imgs):
            break

        # quick val（保留你原来的逻辑）
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)

        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)

        # 解包
        rgb      = datas[0]   # (B,3,H,W)
        da_disp  = datas[1]   # (B,1,H,W) depth-anything disp / relative disp
        lidar    = datas[2]   # (B,1,H,W)
        K_cam    = datas[3]   # (B,3,3)
        gt_depth = datas[4]   # (B,1,H,W)

        if (not printed_data_info) and run.rank == 0:
            print("\n[Train Data Check]")
            print(f"  RGB:        shape={rgb.shape},  range=[{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"  DA disp:    shape={da_disp.shape}, range=[{da_disp.min():.3f}, {da_disp.max():.3f}]")
            print(f"  LiDAR:      shape={lidar.shape},   range=[{lidar.min():.3f}, {lidar.max():.3f}]")
            print(f"  K_cam:      shape={K_cam.shape}")
            print(f"  GT depth:   shape={gt_depth.shape}, range=[{gt_depth.min():.3f}, {gt_depth.max():.3f}]\n")
            if max_train_imgs is not None:
                print(f"[Train] Using at most {max_train_imgs} images per rank this epoch.")
            printed_data_info = True

        # ----------------- Forward -----------------
        # output = run.net(*datas[:-1])
        # ----------------- Forward -----------------
        # 模型 forward = (I, S, K)，不要把 da_disp 传进去
        output = run.net(rgb, lidar, K_cam)

        # ----------------- Supervised loss（GT） -----------------
        sup_loss = run.criterion(output, gt_depth)

        # ----------------- Teacher: disp -> depth proxy (方案B) -----------------
        # 只用 sparse GT 有效点做标定（阈值你可以按单位调整：m 用 1e-3，mm 用 >0）
        gt_mask = (gt_depth > 0)

        with torch.no_grad():
            teacher_depth = disp_to_depth_proxy(
                da_disp, gt_depth, gt_mask,
                eps=1e-6, min_points=50
            )

        # ----------------- SSI distill（真正的 SSI） -----------------
        distill_loss = compute_ssi_loss(output, teacher_depth, valid_mask=None)

        # ----------------- Residual gradient reg（论文风格） -----------------
        grad_loss = compute_reg_loss(output, teacher_depth, valid_mask=None)

        # ------- logging scalar -------
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

        # ----------------- Backward loss 组合（保持你原来的多尺度写法） -----------------
        if isinstance(sup_loss, (list, tuple)):
            loss_list = []
            for i, ls in enumerate(sup_loss):
                ld = distill_loss[i] if isinstance(distill_loss, (list, tuple)) else distill_loss
                lg = grad_loss[i]    if isinstance(grad_loss, (list, tuple))    else grad_loss
                loss_list.append(ls + lambda_distill * ld + lambda_grad * lg)
            loss = loss_list
            sum(loss).backward()
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

        # ----------------- TensorBoard（保持原版逻辑） -----------------
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            if isinstance(loss, (list, tuple)):
                run.writer.add_scalars("Loss", {f"{idx}": l.item() for idx, l in enumerate(loss)}, run.iter)
            else:
                run.writer.add_scalars("Loss", {"0": loss.item()}, run.iter)

            if run.clip and (grad_norm is not None):
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)

        # ----------------- wandb：train scalar log -----------------
        if (run.rank == 0) and (wandb.run is not None) and (run.iter % run.cfg.vis_iter == 0):
            wandb.log({
                "train/lr": run.optimizer.param_groups[0]['lr'],
                "train/loss_supervised": sup_total,
                "train/loss_ssi_raw": distill_total,
                "train/loss_reg_raw": grad_total,
                "train/loss_total": total_for_log,
                "train/iter": run.iter,
                "epoch": run.epoch,
            })

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

# ---------------------------------------------------------------------
# 验证：兼容 RMSE / MetricALL，记录所有 metrics + 可视化 + best checkpoint
# ---------------------------------------------------------------------
def test(run, iter=False):
    net = run.net_ema.module      # EMA 模型用于验证/保存
    best_metric_name = "best_metric_ema"
    net.eval()

    metric_meters = None
    metric_names = getattr(run.metric, "metric_name", None)
    vis_done = False

    max_val_imgs = get_max_images(run.cfg, "max_val_images")
    num_val_imgs = 0
    vis_samples = []

    with torch.no_grad():
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True,
                          leave=False, disable=run.rank):
            if (max_val_imgs is not None) and (num_val_imgs >= max_val_imgs):
                break

            datas = run.init_cuda(*datas)
            # datas: [rgb, da_disp, lidar, K_cam, gt]
            # output = net(datas[0], datas[2], datas[3])
            rgb, da_disp, lidar, K_cam, gt_depth = datas
            output = net(rgb, lidar, K_cam)
            # output = net(*datas[:-1])

            # 单尺度输出用于 metric / 可视化
            if isinstance(output, (list, tuple)):
                output_vis = output[-1]
            else:
                output_vis = output

            # --------- 调用 metric（MetricALL 现在会包含 PSNR/SSIM/MS-SSIM） ---------
            metrics_out = run.metric(output_vis, datas[-1])
            if not isinstance(metrics_out, (list, tuple)):
                metrics_out = [metrics_out]

            # 初始化 meter
            if metric_meters is None:
                metric_meters = [AverageMeter() for _ in range(len(metrics_out))]
                if (metric_names is None) or (len(metric_names) != len(metrics_out)):
                    metric_names = [f"metric_{i}" for i in range(len(metrics_out))]

            # 聚合每个 metric
            for i, m in enumerate(metrics_out):
                if isinstance(m, torch.Tensor):
                    v = m.clone().float()
                else:
                    v = torch.tensor(m, device=output_vis.device, dtype=torch.float32)

                # 兼容 [B,1,1] / [B] / scalar
                if v.dim() > 0:
                    v = v.mean()

                if run.ddp:
                    dist.reduce(v, 0, dist.ReduceOp.AVG)

                metric_meters[i].update(v.item())

            # 更新验证图像计数
            rgb_batch = datas[0]
            num_val_imgs += rgb_batch.shape[0]

            # ----------------- 可视化（跨 batch 累积到 8 张再画一次） -----------------
            if (not iter) and (run.rank == 0) and (not vis_done):
                rgb      = datas[0]      # (B,3,H,W)
                da_disp  = datas[1]      # (B,1,H,W)
                lidar    = datas[2]      # (B,1,H,W)
                gt_depth = datas[-1]     # (B,1,H,W)
                pred     = output_vis    # (B,1,H,W)

                # 用和训练一致的 teacher：disp -> depth proxy
                gt_mask = (gt_depth > 0)
                teacher_depth = disp_to_depth_proxy(
                    da_disp, gt_depth, gt_mask, eps=1e-6, min_points=50
                )

                B = rgb.shape[0]
                for b in range(B):
                    vis_samples.append({
                        "rgb":      rgb[b].detach().cpu(),
                        "da_disp":  da_disp[b].detach().cpu(),
                        "lidar":    lidar[b].detach().cpu(),
                        "gt_depth": gt_depth[b].detach().cpu(),
                        "pred":     pred[b].detach().cpu(),
                        "teacher":  teacher_depth[b].detach().cpu(),
                    })

                if len(vis_samples) >= 8:
                    n_examples = 8
                    fig, axs = plt.subplots(n_examples, 5, figsize=(24, 2.5 * n_examples))
                    if n_examples == 1:
                        axs = np.expand_dims(axs, axis=0)

                    row_titles = [f"Example {i+1}" for i in range(n_examples)]
                    col_titles = ["RGB", "LiDAR", "Teacher(depth-proxy)", "GT Depth", "Pred"]

                    for row in range(n_examples):
                        sample = vis_samples[row]
                        rgb_i      = sample["rgb"].permute(1, 2, 0).numpy()
                        lidar_i    = sample["lidar"][0].numpy()
                        gt_i       = sample["gt_depth"][0].numpy()
                        pred_i     = sample["pred"][0].numpy()
                        teacher_i  = sample["teacher"][0].numpy()

                        # RGB normalize
                        rgb_i = (rgb_i - rgb_i.min()) / (rgb_i.max() - rgb_i.min() + 1e-6)

                        vis_lidar = colorize_depth(lidar_i, vmin=0, vmax=80, cmap='turbo')

                        # Teacher / GT / Pred 可视化建议统一一个范围，方便直观比较
                        vis_teacher = colorize_depth(teacher_i, vmin=0, vmax=80, cmap='turbo')
                        vis_gt      = colorize_depth(gt_i, vmin=0, vmax=80, cmap='turbo')

                        # pred 用百分位避免少量离群点把图压扁（你原来这么做我保留）
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

                    for i, row_t in enumerate(row_titles):
                        axs[i, 0].text(
                            -0.05, 0.5, row_t,
                            transform=axs[i, 0].transAxes,
                            ha="right", va="center", fontsize=10
                        )

                    plt.tight_layout()

                    if wandb.run is not None:
                        wandb.log({"val/vis_8examples_5panels": wandb.Image(fig)}, commit=False)

                    plt.close(fig)
                    vis_done = True

    # ----------------- 写入 TensorBoard & 选择 best -----------------
    if metric_meters is None:
        return

    legand = 'net_ema'
    top1_avg = metric_meters[0].avg
    best_metric_value = getattr(run, best_metric_name, float('inf'))

    # TensorBoard：把所有 metric 都写进去（含 PSNR/SSIM/MS-SSIM）
    if iter:
        for name, meter in zip(metric_names, metric_meters):
            run.writer.add_scalars(f"{name}_Iter", {legand: meter.avg}, run.iter)
    else:
        for name, meter in zip(metric_names, metric_meters):
            run.writer.add_scalars(f"{name}", {legand: meter.avg}, run.epoch)

    # ----------------- 只在非 iter 验证时才考虑更新 best -----------------
    if not iter:
        if run.rank == 0:
            run.ddp_cout(
                f"[VAL] Epoch {run.epoch} primary metric={top1_avg} "
                f"(prev best={best_metric_value})"
            )

        if not math.isfinite(top1_avg):
            if run.rank == 0:
                run.ddp_cout(
                    f"[VAL] Epoch {run.epoch} primary metric is non-finite "
                    f"({top1_avg}), skip best update."
                )
        else:
            if not math.isfinite(best_metric_value):
                is_new_best = True
            else:
                is_new_best = top1_avg < best_metric_value

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

                    run.ddp_cout(
                        f"Epoch {run.epoch} NEW BEST {best_metric_name}: {top1_avg:.7f}\n"
                        f"  -> EMA checkpoint saved to: {ckpt_path}"
                    )

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
                        log_dict = {
                            "best/epoch": run.epoch,
                            "best/metric0": top1_avg,
                            "best/ckpt_path": ckpt_path,
                        }
                        for name, val in best_metrics.items():
                            log_dict[f"best/{name}"] = val
                        wandb.log(log_dict, commit=False)
            else:
                if run.rank == 0:
                    run.ddp_cout(
                        f"Epoch {run.epoch} current: {top1_avg:.7f}, "
                        f"best: {best_metric_value:.7f}.\n"
                    )

    # ----------------- wandb：记录所有 depth metrics（保持你原来的逻辑） -----------------
    if (run.rank == 0) and (wandb.run is not None):
        if iter:
            log_dict = {"iter": run.iter, "epoch": run.epoch}
            for name, meter in zip(metric_names, metric_meters):
                log_dict[f"val_iter/{name}"] = meter.avg
            wandb.log(log_dict)
        else:
            log_dict = {"epoch": run.epoch}
            for name, meter in zip(metric_names, metric_meters):
                log_dict[f"val/{name}"] = meter.avg
            wandb.log(log_dict)


# ---------------------------------------------------------------------
# main：包一层 wandb.init / finish，其余保持原有训练逻辑
# ---------------------------------------------------------------------
@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        # 是否启用 wandb（可通过 ++use_wandb=false 关闭）
        enable_wandb = OmegaConf.select(cfg, "use_wandb", default=True)

        # 只在 rank0 初始化 wandb
        if (run.rank == 0) and enable_wandb:
            run_name = f"{cfg.get('name', 'BPNet')}_{datetime.now().strftime('%m%d_%H%M')}"
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="kitti-dmd3c-bpnet",
                name=run_name,
                config=cfg_dict,
            )

            # 打印数据量方便 sanity check
            try:
                n_train = len(run.trainloader.dataset)
                n_val = len(run.testloader.dataset)
            except Exception:
                n_train = len(run.trainloader)
                n_val = len(run.testloader)
            print(f"[DATA] num_train = {n_train}, num_val = {n_val}")

        # 正式训练：使用 full dataset 或受 max_*_images 控制
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch),
                          desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)

            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()

            # ----------------- 每 2 个 epoch 额外保存一次（按 1-based epoch 计数） -----------------
            epoch_idx = epoch + 1  # 1-based
            if epoch_idx % 2 == 0:
                filename = f"epoch_{epoch_idx:03d}_ema.pth"
                run.save_state(filename=filename)

                if run.rank == 0:
                    base_dir = get_original_cwd()
                    ckpt_dir = os.path.join(base_dir, 'checkpoint', run.cfg.name)
                    run.ddp_cout(
                        f"[Epoch {epoch_idx}] Periodic checkpoint saved to: "
                        f"{os.path.join(ckpt_dir, filename)}"
                    )
        # ----------------- 训练结束：打印 / 写入最终 best 信息 -----------------
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
                print("\n[WARN] No validation metrics were computed; "
                      "best checkpoint was never updated.")

            # 同步到 wandb summary
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
