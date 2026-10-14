# # -*- coding: utf-8 -*-
# # @File : train_amp.py
# # @Project: BP-Net
# # @Author : jie
# # @Time : 10/27/21 3:58 PM

# import torch
# from tqdm import tqdm
# import hydra
# import torch.distributed as dist
# from utils import *


# def train(run):
#     for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
#         if run.epoch >= run.cfg.test_epoch:
#             if run.iter % run.cfg.test_iter == 0:
#                 test(run, iter=True)
#         datas = run.init_cuda(*datas)
#         run.net.train()
#         run.optimizer.zero_grad(set_to_none=True)
#         output = run.net(*datas[:-1])
#         loss = run.criterion(output, datas[-1])
#         sum(loss).backward()
#         if run.clip:
#             grad_norm = run.clip(run.net.parameters())
#         run.optimizer.step()
#         run.net_ema.update(run.net)
#         if run.lr_iter:
#             run.lr_scheduler.step()
#         if run.iter % run.cfg.vis_iter == 0:
#             print(f"[Iter {run.iter}] Loss: {[l.item() for l in loss]}, LR: {run.optimizer.param_groups[0]['lr']}")

#         if run.iter % run.cfg.vis_iter == 0:
#             run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
#             run.writer.add_scalars("Loss", {f"{idx}": l.item() for idx, l in enumerate(loss)}, run.iter)
#             if run.clip and (grad_norm is not None):
#                 run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)
#         run.iter += 1
#     if not run.lr_iter:
#         run.lr_scheduler.step()
#     run.writer.flush()


# def test(run, iter=False):
#     top1 = AverageMeter()
#     net = run.net_ema.module
#     best_metric_name = "best_metric_ema"
#     legand = 'net_ema'
#     net.eval()
#     with torch.no_grad():
#         for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
#             datas = run.init_cuda(*datas)
#             output = net(*datas[:-1])
#             if isinstance(output, (list, tuple)):
#                 output = output[-1]
#             prec1 = run.metric(output, datas[-1])
#             if isinstance(prec1, (list, tuple)):
#                 prec1 = prec1[0]
#             if run.ddp:
#                 dist.reduce(prec1, 0, dist.ReduceOp.AVG)
#             top1.update(prec1.item())
#     if iter:
#         run.writer.add_scalars("RMSE_Iter", {legand: top1.avg}, run.iter)
#     else:
#         run.writer.add_scalars("RMSE", {legand: top1.avg}, run.epoch)
#     if top1.avg < getattr(run, best_metric_name):
#         setattr(run, best_metric_name, top1.avg)
#         run.save_state()
#         run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')


# @hydra.main(config_path='configs', config_name='config', version_base='1.2')
# def main(cfg):
#     with Trainer(cfg) as run:
#         for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
#             run.epoch = epoch
#             if run.train_sampler:
#                 run.train_sampler.set_epoch(epoch)
#             train(run)
#             torch.cuda.synchronize()
#             test(run)
#             torch.cuda.synchronize()


# if __name__ == '__main__':
#     main()

# -*- coding: utf-8 -*-
# @File : train_amp_friendly.py
# @Project: BP-Net
# @Author : jie + modified by ChatGPT
# @Time : 2025-10-08

import torch
from tqdm import tqdm
import hydra
import torch.distributed as dist
from utils import *

def train(run):
    for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)
        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)
        output = run.net(*datas[:-1])
        loss = run.criterion(output, datas[-1])
        sum(loss).backward()
        grad_norm = None
        if run.clip:
            grad_norm = run.clip(run.net.parameters())
        run.optimizer.step()
        run.net_ema.update(run.net)
        if run.lr_iter:
            run.lr_scheduler.step()

        # TensorBoard 记录 & terminal 输出
        if run.iter % run.cfg.vis_iter == 0:
            lr_val = run.optimizer.param_groups[0]['lr']
            loss_vals = [l.item() for l in loss]
            print(f"[Iter {run.iter}] Loss: {loss_vals}, LR: {lr_val}")
            run.writer.add_scalar("Lr", lr_val, run.iter)
            run.writer.add_scalars("Loss", {f"{idx}": l for idx, l in enumerate(loss_vals)}, run.iter)
            if run.clip and (grad_norm is not None):
                print(f"GradNorm: {grad_norm.item()}")
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)

        run.iter += 1

    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()

def test(run, iter=False):
    top1 = AverageMeter()
    net = run.net_ema.module
    best_metric_name = "best_metric_ema"
    legand = 'net_ema'
    net.eval()
    with torch.no_grad():
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            datas = run.init_cuda(*datas)
            output = net(*datas[:-1])
            if isinstance(output, (list, tuple)):
                output = output[-1]
            prec1 = run.metric(output, datas[-1])
            if isinstance(prec1, (list, tuple)):
                prec1 = prec1[0]
            if run.ddp:
                dist.reduce(prec1, 0, dist.ReduceOp.AVG)
            top1.update(prec1.item())

    # terminal 输出 RMSE
    if iter:
        print(f"[Iter {run.iter}] RMSE: {top1.avg:.4f}")
        run.writer.add_scalars("RMSE_Iter", {legand: top1.avg}, run.iter)
    else:
        print(f"[Epoch {run.epoch}] RMSE: {top1.avg:.4f}")
        run.writer.add_scalars("RMSE", {legand: top1.avg}, run.epoch)

    # checkpoint 只保存最好的 EMA 权重
    if top1.avg < getattr(run, best_metric_name):
        setattr(run, best_metric_name, top1.avg)
        run.save_state()
        run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)
            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()

if __name__ == '__main__':
    main()
