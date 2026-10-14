# # -*- coding: utf-8 -*-
# # @File   : test.py
# # @Project: BP-Net (+ DMD3C)
# # @Desc   : 使用与 train_amp 相同的 pipeline 做 val/test，
# #           计算与训练时一致的 metrics，并保存 16bit 单通道预测。

# import os
# import torch
# from tqdm import tqdm
# import hydra
# from PIL import Image
# from omegaconf import OmegaConf
# import torch.distributed as dist

# from utils import *   # Trainer, AverageMeter, etc.


# def evaluate(run, mode=None, save=False, save_dir=None):
#     """
#     与 train_amp.py 里的 test() 在核心逻辑上保持一致：
#     - 使用 run.testloader（其中已经包含所有预处理和 crop）
#     - 使用 run.metric 做 RMSE / MetricALL 计算
#     - 多 GPU 情况下用 dist.reduce 做平均
#     - 可选保存 16bit 单通道 PNG（pred_depth * 256）

#     参数：
#         run     : Trainer 实例
#         mode    : 仅用于日志和默认保存路径；如果为 None，则用 cfg.data.testset.mode
#         save    : 是否保存预测 PNG
#         save_dir: 保存目录；如果为 None，则用 results/{run.cfg.name}/{mode}
#     """
#     dataloader = run.testloader
#     net = run.net_ema.module
#     net.eval()

#     import inspect
#     # forward(self, ...) 里除了 self 之外需要的输入个数
#     n_in = len(inspect.signature(net.forward).parameters) - 1
#     if run.rank == 0:
#         run.ddp_cout(f"[DEBUG] net.forward signature = {inspect.signature(net.forward)}")
#         run.ddp_cout(f"[DEBUG] net expects {n_in} inputs (excluding self)")


#     # 确定模式（仅用于路径命名）
#     if mode is None:
#         mode = getattr(run.cfg.data.testset, "mode", "selval")

#     # metric 名称与个数（比如 RMSE 或 MetricALL 里的 11 个指标）
#     metric_names = getattr(run.metric, "metric_name", ["RMSE"])
#     meters = [AverageMeter() for _ in range(len(metric_names))]

#     # 打印一下样本数量，防止 num_test=0 这种情况
#     if run.rank == 0:
#         try:
#             num_test = len(dataloader.dataset)
#         except Exception:
#             num_test = -1
#         run.ddp_cout(f"[TEST] mode={mode}, num_test={num_test}")

#     # 决定保存目录
#     if save:
#         if save_dir is None:
#             save_dir = os.path.join("results", run.cfg.name, mode)
#         if run.rank == 0:
#             os.makedirs(save_dir, exist_ok=True)
#             run.ddp_cout(f"[TEST] Saving predictions to: {save_dir}")

#     with torch.no_grad():
#         for idx, datas in enumerate(
#                 tqdm(dataloader, desc="test ", dynamic_ncols=True,
#                      leave=False, disable=run.rank)):
#             # 和训练完全一致：先做 cuda / DDP 封装
#             datas = run.init_cuda(*datas)

#             # 前向：按 forward 实际需要的输入个数来喂，避免 dataset 多返回 K_cam / meta 导致参数数量不匹配
#             inputs = datas[:-1]                 # 最后一个仍然是 GT
#             inputs = inputs[:n_in]              # 只取前 n_in 个输入

#             # 可选：第一次打印一下 datas 长度，确认到底多了什么（只在 rank0 打印一次）
#             if run.rank == 0 and idx == 0:
#                 run.ddp_cout(f"[DEBUG] len(datas)={len(datas)}; using len(inputs)={len(inputs)}")
#                 run.ddp_cout(f"[DEBUG] shapes={[getattr(x,'shape',None) for x in datas]}")

#             output = net(*inputs)

#             # # 前向：保持 *datas[:-1]，保证与训练/val 相同的输入结构
#             # output = net(*datas[:-1])
#             if isinstance(output, (list, tuple)):
#                 pred = output[-1]      # 单尺度输出用于 metric 和保存
#             else:
#                 pred = output          # (B,1,H,W)

#             # ----------------- 计算 metrics（与 train_amp 里的 test 一致） -----------------
#             metrics_out = run.metric(pred, datas[-1])   # datas[-1] 就是 GT depth
#             if not isinstance(metrics_out, (list, tuple)):
#                 metrics_out = [metrics_out]

#             for i, m in enumerate(metrics_out):
#                 if isinstance(m, torch.Tensor):
#                     v = m.clone().float()
#                 else:
#                     v = torch.tensor(m, device=pred.device, dtype=torch.float32)

#                 # 兼容 RMSE 标量 / MetricALL 的 [B,1,1] / [B]
#                 if v.dim() > 0:
#                     v = v.mean()

#                 if run.ddp:
#                     dist.reduce(v, 0, dist.ReduceOp.AVG)

#                 meters[i].update(v.item())

#             # ----------------- 保存 16bit 单通道预测（KITTI 风格） -----------------
#             if save and (run.rank == 0):
#                 batch_size = pred.shape[0]
#                 for b in range(batch_size):
#                     global_index = idx * batch_size + b

#                     # 如果 Dataset 定义了 names，则用 names 保持与 GT 对齐
#                     if hasattr(dataloader.dataset, "names") and \
#                             global_index < len(dataloader.dataset.names):
#                         base_name = dataloader.dataset.names[global_index]
#                     else:
#                         base_name = f"{global_index:010d}.png"

#                     out_path = os.path.join(save_dir, base_name)

#                     # pred 是以 meter 为单位的浮点深度，和训练/val 保持一致
#                     depth = pred[b, 0].detach().cpu().numpy()
#                     # 按 KITTI 约定：depth(m) * 256 -> uint16
#                     depth_16 = (depth * 256.0).astype('uint16')

#                     Image.fromarray(depth_16).save(out_path)

#     # ----------------- 打印 / 记录 metrics（与 train_amp.test 风格一致） -----------------
#     log_str = ""
#     for name, meter in zip(metric_names, meters):
#         log_str += f" {name}:{meter.avg:.7f} "
#     run.ddp_log(log_str, always=False)


# @hydra.main(config_path='configs', config_name='config', version_base='1.2')
# def main(cfg):
#     """
#     入口：
#     - 仍然使用 configs/config.yaml
#     - 使用 Trainer(cfg) 创建与训练完全一致的 run（包含 net_ema, metric, dataloader 等）
#     - 通过命令行 ++save / ++save_dir 控制是否保存预测与保存位置
#     """
#     mode = OmegaConf.select(cfg, "data.testset.mode", default="selval")
#     save = OmegaConf.select(cfg, "save", default=True)
#     save_dir = OmegaConf.select(cfg, "save_dir", default=None)


#     with Trainer(cfg) as run:
#         evaluate(run, mode=mode, save=save, save_dir=save_dir)


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
# @File   : test.py
# @Project: BP-Net (+ DMD3C)
# @Desc   : 使用与 train_amp 相同的 pipeline 做 val/test，
#           计算与训练时一致的 metrics，并保存 16bit 单通道预测。
# -*- coding: utf-8 -*-
# @File   : test.py
# @Project: BP-Net (+ DMD3C)
# @Desc   : 使用与 train_amp 相同的 pipeline 做 val/test，
#           计算与训练时一致的 metrics，并保存 16bit 单通道预测。

import os
import sys
import inspect
import torch
from tqdm import tqdm
import hydra
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist

# 确保 BP-Net 目录在 sys.path 末尾，使 models 包可被 Hydra 找到，
# 同时保证 DMD3C/ 本地模块（datasets、augs 等）优先级更高
_bp_net_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _bp_net_dir not in sys.path:
    sys.path.append(_bp_net_dir)

from utils import *   # Trainer, AverageMeter, etc.


def _zero_fraction(x: torch.Tensor) -> float:
    """估计稀疏程度：0 的比例（越大越稀疏）。"""
    if (not torch.is_tensor(x)) or x.numel() == 0:
        return -1.0
    # 注意：这里仅用于 heuristic，速度够用
    return (x == 0).float().mean().item()


def _pick_inputs_for_forward(datas, net, run=None):
    """
    根据 forward 参数名 + datas 的 shape，自动挑选 (I, S, K) 输入。
    datas 通常是: [RGB(B,3,H,W), DA_disp(B,1,H,W), LiDAR(B,1,H,W), K(B,3,3), GT(B,1,H,W)]
    """
    # 1) 拿到 forward 的参数名（bound method 一般不含 self）
    sig = inspect.signature(net.forward)
    param_names = [p.name for p in sig.parameters.values()]
    if param_names and param_names[0] == "self":
        param_names = param_names[1:]

    # 2) 从 datas 里识别 RGB / K / 候选深度图
    rgb = None
    K = None
    depth_maps = []  # (tensor, idx)

    for i, d in enumerate(datas):
        if not torch.is_tensor(d):
            continue
        if d.dim() == 4 and d.shape[1] == 3:
            rgb = d
        elif d.dim() == 3 and d.shape[-2:] == (3, 3):
            K = d
        elif d.dim() == 4 and d.shape[1] == 1:
            depth_maps.append((d, i))

    # 3) S：从所有 (B,1,H,W) 里选最稀疏的那个（通常 LiDAR）
    S = None
    if depth_maps:
        # 排除最后一个 GT（通常也是 B,1,H,W）——但不依赖它就是最后一个，也尽量排除 datas[-1]
        candidates = []
        for t, idx in depth_maps:
            if idx == len(datas) - 1:
                continue
            candidates.append((t, idx))

        if not candidates:
            candidates = depth_maps

        # 选 zero_fraction 最大者
        candidates = sorted(candidates, key=lambda x: _zero_fraction(x[0]), reverse=True)
        S = candidates[0][0]

    # 4) 根据 forward 的参数名顺序组装 inputs
    # 你的打印显示 forward signature = (I, S, K)，所以这里会组装 [rgb, S, K]
    name_to_tensor = {
        "I": rgb, "img": rgb, "image": rgb, "rgb": rgb, "x": rgb,
        "S": S, "s": S, "sparse": S, "lidar": S, "depth": S,
        "K": K, "k": K, "intr": K, "intrinsics": K, "K_cam": K,
    }

    inputs = []
    for n in param_names:
        if n in name_to_tensor and name_to_tensor[n] is not None:
            inputs.append(name_to_tensor[n])
        else:
            # 如果参数名不在字典里，就按最合理的默认：
            # 先填 rgb，再填 S，再填 K
            if rgb is not None and rgb not in inputs and len(inputs) < len(param_names):
                inputs.append(rgb)
                continue
            if S is not None and S not in inputs and len(inputs) < len(param_names):
                inputs.append(S)
                continue
            if K is not None and K not in inputs and len(inputs) < len(param_names):
                inputs.append(K)
                continue

    # 5) 校验：如果 forward 要 3 个，但 inputs 不足，就报一个更清楚的错误
    if len(inputs) != len(param_names):
        msg = (
            f"Cannot build inputs for net.forward{sig}. "
            f"param_names={param_names}, got_inputs={len(inputs)}. "
            f"Detected rgb={rgb is not None}, S={S is not None}, K={K is not None}."
        )
        raise RuntimeError(msg)

    # debug 打印一次
    if run is not None and run.rank == 0:
        run.ddp_cout(f"[DEBUG] forward signature = {sig}")
        run.ddp_cout(f"[DEBUG] forward param_names = {param_names}")
        run.ddp_cout(f"[DEBUG] picked inputs shapes = {[getattr(x,'shape',None) for x in inputs]}")

    return inputs


def evaluate(run, mode=None, save=False, save_dir=None):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()

    # 确定模式（仅用于路径命名）
    if mode is None:
        mode = getattr(run.cfg.data.testset, "mode", "selval")

    # metric 名称与个数
    metric_names = getattr(run.metric, "metric_name", ["RMSE"])
    meters = [AverageMeter() for _ in range(len(metric_names))]

    # 打印一下样本数量
    if run.rank == 0:
        try:
            num_test = len(dataloader.dataset)
        except Exception:
            num_test = -1
        run.ddp_cout(f"[TEST] mode={mode}, num_test={num_test}")

    # 决定保存目录
    if save:
        if save_dir is None:
            save_dir = os.path.join("results", run.cfg.name, mode)
        if run.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            run.ddp_cout(f"[TEST] Saving predictions to: {save_dir}")

    printed_once = False

    with torch.no_grad():
        for idx, datas in enumerate(
                tqdm(dataloader, desc="test ", dynamic_ncols=True,
                     leave=False, disable=run.rank)):

            # cuda / DDP 封装
            datas = run.init_cuda(*datas)

            # GT 默认 datas[-1]
            gt = datas[-1]

            # 只在第一个 batch 打印一次 datas 结构
            if run.rank == 0 and (not printed_once):
                shapes = [getattr(x, "shape", None) for x in datas]
                run.ddp_cout(f"[DEBUG] len(datas)={len(datas)}")
                run.ddp_cout(f"[DEBUG] datas shapes = {shapes}")
                printed_once = True

            # 关键：按 forward(I,S,K) 选输入（不把 DA_disp 传进网络）
            inputs = _pick_inputs_for_forward(datas, net, run=run if (run.rank == 0 and idx == 0) else None)

            # 前向
            output = net(*inputs)

            # 单尺度输出用于 metric 和保存
            if isinstance(output, (list, tuple)):
                pred = output[-1]
            else:
                pred = output

            # ----------------- 计算 metrics -----------------
            metrics_out = run.metric(pred, gt)
            if not isinstance(metrics_out, (list, tuple)):
                metrics_out = [metrics_out]

            for i, m in enumerate(metrics_out):
                if isinstance(m, torch.Tensor):
                    v = m.clone().float()
                else:
                    v = torch.tensor(m, device=pred.device, dtype=torch.float32)

                if v.dim() > 0:
                    v = v.mean()

                if run.ddp:
                    dist.reduce(v, 0, dist.ReduceOp.AVG)

                meters[i].update(v.item())

            # ----------------- 保存 16bit 单通道预测（KITTI 风格） -----------------
            if save and (run.rank == 0):
                batch_size = pred.shape[0]
                for b in range(batch_size):
                    global_index = idx * batch_size + b

                    if hasattr(dataloader.dataset, "names") and global_index < len(dataloader.dataset.names):
                        base_name = dataloader.dataset.names[global_index]
                    else:
                        base_name = f"{global_index:010d}.png"

                    out_path = os.path.join(save_dir, base_name)

                    depth = pred[b, 0].detach().cpu().numpy()
                    depth_16 = (depth * 256.0).astype("uint16")
                    Image.fromarray(depth_16).save(out_path)

    # ----------------- 打印 / 记录 metrics -----------------
    log_str = ""
    for name, meter in zip(metric_names, meters):
        log_str += f" {name}:{meter.avg:.7f} "
    run.ddp_log(log_str, always=False)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    mode = OmegaConf.select(cfg, "data.testset.mode", default="selval")
    save = OmegaConf.select(cfg, "save", default=True)
    save_dir = OmegaConf.select(cfg, "save_dir", default=None)

    with Trainer(cfg) as run:
        evaluate(run, mode=mode, save=save, save_dir=save_dir)


if __name__ == "__main__":
    main()
