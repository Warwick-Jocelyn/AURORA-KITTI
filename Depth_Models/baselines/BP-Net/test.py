# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
from tqdm import tqdm
import hydra
from PIL import Image
import os
from omegaconf import OmegaConf
from utils import *


# def test(run, mode='selval', save=True):
#     dataloader = run.testloader
#     net = run.net_ema.module
#     net.eval()
#     tops = [AverageMeter() for i in range(len(run.metric.metric_name))]
#     if save:
#         dir_path = f'results/{run.cfg.name}/{mode}'
#         os.makedirs(dir_path, exist_ok=True)
#     with torch.no_grad():
#         for idx, datas in enumerate(
#                 tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
#             datas = run.init_cuda(*datas)
#             output = net(*datas[:-1])
#             if isinstance(output, (list, tuple)):
#                 output = output[-1]
#             precs = run.metric(output, datas[-1])
#             for prec, top in zip(precs, tops):
#                 top.update(prec.mean().detach().cpu().item())
#             if save:
#                 for i in range(output.shape[0]):
#                     index = idx * output.shape[0] + i
#                     file_path = os.path.join(dir_path, f'{index:010d}.png')
#                     img = (output[i, 0] * 256.0).detach().cpu().numpy().astype('uint16')
#                     Img = Image.fromarray(img)
#                     Img.save(file_path)
#     logs = ""
#     for name, top in zip(run.metric.metric_name, tops):
#         logs += f" {name}:{top.avg:.7f} "
#     run.ddp_log(logs, always=True)
def test(run, mode='selval', save=True):
    dataloader = run.testloader
    dataset = dataloader.dataset          # 关键：拿到原始 Dataset
    net = run.net_ema.module
    net.eval()
    tops = [AverageMeter() for i in range(len(run.metric.metric_name))]
    if save:
        dir_path = f'results/{run.cfg.name}/{mode}'
        os.makedirs(dir_path, exist_ok=True)

    global_index = 0  # 全局样本 index，从 0 开始

    with torch.no_grad():
        for idx, datas in enumerate(
                tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
            datas = run.init_cuda(*datas)
            output = net(*datas[:-1])
            if isinstance(output, (list, tuple)):
                output = output[-1]

            # 计算 metric
            precs = run.metric(output, datas[-1])
            for prec, top in zip(precs, tops):
                top.update(prec.mean().detach().cpu().item())

            if save:
                batch_size = output.shape[0]
                for i in range(batch_size):
                    # --- 用原始文件名 ---
                    # dataset.names 是在 KITTI.__init__ 里根据 self.depths 定的
                    orig_name = dataset.names[global_index]      # e.g. 2011_..._image_02.png
                    base_name, _ = os.path.splitext(orig_name)   # 去掉后缀
                    save_name = base_name + '.png'               # 确保是 .png

                    file_path = os.path.join(dir_path, save_name)

                    img = (output[i, 0] * 256.0).detach().cpu().numpy().astype('uint16')
                    Img = Image.fromarray(img)
                    Img.save(file_path)

                    global_index += 1  # 下一个样本

    logs = ""
    for name, top in zip(run.metric.metric_name, tops):
        logs += f" {name}:{top.avg:.7f} "
    run.ddp_log(logs, always=True)


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        test(run, mode=cfg.data.testset.mode, save=OmegaConf.select(cfg, 'save', default=True))



if __name__ == '__main__':
    main()
