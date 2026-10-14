# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# # @Filename:    criteria.py
# # @Project:     BP-Net
# # @Author:      jie
# # @Time:        2021/3/14 7:51 PM

# import torch
# import torch.nn as nn

# __all__ = [
#     'RMSE',
#     'MSMSE',
#     'MetricALL',
# ]


# class RMSE(nn.Module):

#     def __init__(self, mul_factor=1.):
#         super().__init__()
#         self.mul_factor = mul_factor
#         self.metric_name = [
#             'RMSE',
#         ]

#     def forward(self, outputs, target):
#         outputs = outputs / self.mul_factor
#         target = target / self.mul_factor
#         val_pixels = (target > 1e-3).float()
#         err = (target * val_pixels - outputs * val_pixels) ** 2
#         loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
#         cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
#         return torch.sqrt(loss / cnt).mean(),


# class MSMSE(nn.Module):
#     """
#     Multi-Scale MSE
#     """

#     def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1)):
#         super().__init__()
#         self.deltas = deltas

#     def mse(self, est, gt):
#         valid = (gt > 1e-3).float()
#         loss = est * valid - gt * valid
#         return (loss ** 2).mean()

#     def forward(self, outputs, target):
#         loss = [delta * self.mse(ests, target) for ests, delta in zip(outputs, self.deltas)]
#         return loss


# class MetricALL(nn.Module):
#     def __init__(self, mul_factor):
#         super().__init__()
#         self.t_valid = 0.0001
#         self.mul_factor = mul_factor
#         self.metric_name = [
#             'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
#         ]

#     def forward(self, pred, gt):
#         with torch.no_grad():
#             pred = pred.detach() / self.mul_factor
#             gt = gt.detach() / self.mul_factor
#             pred_inv = 1.0 / (pred + 1e-8)
#             gt_inv = 1.0 / (gt + 1e-8)

#             # For numerical stability
#             mask = gt > self.t_valid
#             # num_valid = mask.sum()
#             B = mask.size(0)
#             num_valid = torch.sum(mask.view(B, -1), -1, keepdim=True)

#             # pred = pred[mask]
#             # gt = gt[mask]
#             pred = pred * mask
#             gt = gt * mask

#             # pred_inv = pred_inv[mask]
#             # gt_inv = gt_inv[mask]
#             pred_inv = pred_inv * mask
#             gt_inv = gt_inv * mask

#             # pred_inv[pred <= self.t_valid] = 0.0
#             # gt_inv[gt <= self.t_valid] = 0.0

#             # RMSE / MAE
#             diff = pred - gt
#             diff_abs = torch.abs(diff)
#             diff_sqr = torch.pow(diff, 2)

#             rmse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             rmse = torch.sqrt(rmse)

#             mae = torch.sum(diff_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

#             # iRMSE / iMAE
#             diff_inv = pred_inv - gt_inv
#             diff_inv_abs = torch.abs(diff_inv)
#             diff_inv_sqr = torch.pow(diff_inv, 2)

#             irmse = torch.sum(diff_inv_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             irmse = torch.sqrt(irmse)

#             imae = torch.sum(diff_inv_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

#             # Rel
#             rel = diff_abs / (gt + 1e-8)
#             rel = torch.sum(rel.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

#             # delta
#             r1 = gt / (pred + 1e-8)
#             r2 = pred / (gt + 1e-8)
#             ratio = torch.max(r1, r2)

#             ratio = torch.max(ratio, 10000 * (1 - mask.float()))

#             del_1 = (ratio < 1.25).type_as(ratio)
#             del_2 = (ratio < 1.25 ** 2).type_as(ratio)
#             del_3 = (ratio < 1.25 ** 3).type_as(ratio)
#             del_102 = (ratio < 1.02).type_as(ratio)
#             del_105 = (ratio < 1.05).type_as(ratio)
#             del_110 = (ratio < 1.10).type_as(ratio)

#             del_1 = torch.sum(del_1.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             del_2 = torch.sum(del_2.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             del_3 = torch.sum(del_3.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             del_102 = torch.sum(del_102.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             del_105 = torch.sum(del_105.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
#             del_110 = torch.sum(del_110.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

#             result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]

#         return result
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict

__all__ = [
    'RMSE',
    'MSMSE',
    'MetricALL',
]

# -----------------------------
# SSIM / MS-SSIM helpers (PyTorch)
# -----------------------------
def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    return g

def _create_gaussian_window(window_size: int, sigma: float, device, dtype):
    g1 = _gaussian_1d(window_size, sigma, device, dtype)
    g2 = (g1[:, None] * g1[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return g2

def _ssim_per_image(x, y, data_range: float, window_size=11, sigma=1.5, K1=0.01, K2=0.03):
    """
    x,y: (B,1,H,W) in [0, data_range]
    Returns:
      ssim: (B,1)
      mcs:  (B,1)  (contrast-structure term, used for MS-SSIM)
    """
    B, C, H, W = x.shape
    ws = min(window_size, H, W)
    if ws % 2 == 0:
        ws -= 1
    if ws < 3:
        # too small to compute meaningful SSIM
        ssim = x.new_zeros((B, 1))
        mcs = x.new_zeros((B, 1))
        return ssim, mcs

    window = _create_gaussian_window(ws, sigma, x.device, x.dtype)
    padding = ws // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=1)
    mu_y = F.conv2d(y, window, padding=padding, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=1) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # SSIM
    num1 = 2 * mu_xy + C1
    den1 = mu_x2 + mu_y2 + C1
    num2 = 2 * sigma_xy + C2
    den2 = sigma_x2 + sigma_y2 + C2

    ssim_map = (num1 / (den1 + 1e-12)) * (num2 / (den2 + 1e-12))
    mcs_map = (num2 / (den2 + 1e-12))  # contrast-structure

    ssim = ssim_map.mean(dim=(2, 3), keepdim=False).view(B, 1)
    mcs = mcs_map.mean(dim=(2, 3), keepdim=False).view(B, 1)
    return ssim, mcs

def _ms_ssim_per_image(x, y, data_range: float, weights=None, levels=5):
    """
    x,y: (B,1,H,W) in [0, data_range]
    Returns msssim: (B,1)
    """
    if weights is None:
        # standard 5-level MS-SSIM weights
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = weights[:levels]

    mcs_list = []
    ssim_list = []

    x_i, y_i = x, y
    for i in range(levels):
        ssim_i, mcs_i = _ssim_per_image(x_i, y_i, data_range=data_range)
        ssim_list.append(ssim_i)
        mcs_list.append(mcs_i)

        if i < levels - 1:
            # downsample
            x_i = F.avg_pool2d(x_i, kernel_size=2, stride=2)
            y_i = F.avg_pool2d(y_i, kernel_size=2, stride=2)

    # MS-SSIM = prod(mcs_i^w_i) * ssim_last^w_last  (typical definition)
    msssim = x.new_ones(ssim_list[0].shape)
    for i in range(levels - 1):
        w = weights[i]
        msssim = msssim * torch.clamp(mcs_list[i], min=1e-6) ** w
    msssim = msssim * torch.clamp(ssim_list[-1], min=1e-6) ** weights[-1]
    return msssim


class RMSE(nn.Module):
    def __init__(self, mul_factor=1.):
        super().__init__()
        self.mul_factor = mul_factor
        self.metric_name = ['RMSE']

    def forward(self, outputs, target):
        outputs = outputs / self.mul_factor
        target = target / self.mul_factor
        val_pixels = (target > 1e-3).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt).mean(),


class MSMSE(nn.Module):
    """
    Multi-Scale MSE
    """
    def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1)):
        super().__init__()
        self.deltas = deltas

    def mse(self, est, gt):
        valid = (gt > 1e-3).float()
        loss = est * valid - gt * valid
        return (loss ** 2).mean()

    def forward(self, outputs, target):
        loss = [delta * self.mse(ests, target) for ests, delta in zip(outputs, self.deltas)]
        return loss


class MetricALL(nn.Module):
    def __init__(self, mul_factor, max_depth=80.0):
        super().__init__()
        self.t_valid = 0.0001
        self.mul_factor = mul_factor
        self.max_depth = float(max_depth)

        # 新增 3 个 IQA-like 指标
        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL',
            'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110',
            'PSNR', 'SSIM', 'MS-SSIM'
        ]

    def forward(self, pred, gt):
        with torch.no_grad():
            pred = pred.detach() / self.mul_factor
            gt = gt.detach() / self.mul_factor

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            mask = gt > self.t_valid
            B = mask.size(0)
            num_valid = torch.sum(mask.view(B, -1), -1, keepdim=True)  # (B,1)

            pred_m = pred * mask
            gt_m = gt * mask
            pred_inv_m = pred_inv * mask
            gt_inv_m = gt_inv * mask

            # ---------------- Basic depth metrics ----------------
            diff = pred_m - gt_m
            diff_abs = torch.abs(diff)
            diff_sqr = diff.pow(2)

            rmse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = torch.sum(diff_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            diff_inv = pred_inv_m - gt_inv_m
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = diff_inv.pow(2)

            irmse = torch.sum(diff_inv_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = torch.sum(diff_inv_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            rel = diff_abs / (gt_m + 1e-8)
            rel = torch.sum(rel.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # delta
            r1 = gt_m / (pred_m + 1e-8)
            r2 = pred_m / (gt_m + 1e-8)
            ratio = torch.max(r1, r2)

            ratio = torch.max(ratio, 10000 * (1 - mask.float()))

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = torch.sum(del_1.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_2 = torch.sum(del_2.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_3 = torch.sum(del_3.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_102 = torch.sum(del_102.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_105 = torch.sum(del_105.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_110 = torch.sum(del_110.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # ---------------- New: PSNR / SSIM / MS-SSIM ----------------
            # PSNR: MSE over valid pixels, peak = max_depth
            mse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            peak = pred.new_tensor(self.max_depth)
            psnr = torch.where(
                mse <= 1e-12,
                pred.new_full(mse.shape, 100.0),
                20.0 * torch.log10(peak) - 10.0 * torch.log10(mse + 1e-12)
            )

            # SSIM/MS-SSIM on clipped maps; set invalid pixels to 0 in both
            pred_clip = torch.clamp(pred, 0.0, self.max_depth)
            gt_clip = torch.clamp(gt, 0.0, self.max_depth)
            pred_clip = pred_clip * mask
            gt_clip = gt_clip * mask

            ssim_val, _ = _ssim_per_image(gt_clip, pred_clip, data_range=self.max_depth)
            msssim_val = _ms_ssim_per_image(gt_clip, pred_clip, data_range=self.max_depth)

            result = [
                rmse, mae, irmse, imae, rel,
                del_1, del_2, del_3, del_102, del_105, del_110,
                psnr, ssim_val, msssim_val
            ]

        return result


class DenoiseMetrics:
    """Model-agnostic DCD denoising metrics: ERR(+SR), DCE, NRR(+CPP).

    Design & formulas: DCD_denoising_metrics_design.md.
    Error is ALWAYS computed vs the dense GT `G`; the clean/weather sparse pair
    `(S_c, S_w)` only locates the weather noise (masks). Works on any method's
    predicted dense depth `P`, so it is reusable for DDCD, the modality ablations,
    and the baseline methods.

    Usage (pixel-pooled across the test set):
        dm = DenoiseMetrics(tau=0.05)
        for frame: dm.add(P, G, S_w, S_c, groups=["ALL", weather, (weather, sev)])
        table = dm.reduce_all()          # {group: {ERR, SR, DCE, DCE_mae, NRR, CPP, n_*}}
    All inputs are 2-D float32 arrays in METERS (decode PNGs with the right scale
    before calling: GT/pred ÷256, our sparse sets ÷255 — see the design doc §0).
    """

    def __init__(self, tau=0.05, t_valid=1e-3):
        self.tau = float(tau)            # relative range-perturbation threshold (§3.5)
        self.t_valid = float(t_valid)
        self.acc = defaultdict(self._zero)

    @staticmethod
    def _zero():
        return dict(cor_P_se=0.0, cor_Sw_se=0.0, cor_n=0, cor_sr_hit=0,
                    drop_P_se=0.0, drop_P_ae=0.0, drop_n=0,
                    noise_P_se=0.0, noise_n=0, keep_P_se=0.0, keep_n=0)

    def add(self, P, G, S_w, S_c, groups=("ALL",)):
        """Accumulate one frame into each key in `groups`."""
        P = np.asarray(P, np.float64); G = np.asarray(G, np.float64)
        S_w = np.asarray(S_w, np.float64); S_c = np.asarray(S_c, np.float64)
        Mg = G > self.t_valid
        Mw = S_w > self.t_valid
        Mc = S_c > self.t_valid
        dpert = np.abs(S_c - S_w) > self.tau * np.maximum(S_c, 1e-6)   # relative test
        Mdrop = Mc & ~Mw
        Mspur = Mw & ~Mc
        Mpert = Mc & Mw & dpert
        Mkeep = Mc & Mw & ~dpert
        Wcor = (Mspur | Mpert) & Mg          # input LiDAR present but WRONG
        Wdrop = Mdrop & Mg                   # input LiDAR missing
        Wnoise = (Mdrop | Mspur | Mpert) & Mg
        Wkeep = Mkeep & Mg

        def se(mask, A):
            d = A[mask] - G[mask]
            return float((d * d).sum())

        sr_hit = 0
        if Wcor.any():
            sr_hit = int((np.abs(P[Wcor] - G[Wcor]) < np.abs(S_w[Wcor] - G[Wcor])).sum())
        cor_P_se, cor_Sw_se, cor_n = se(Wcor, P), se(Wcor, S_w), int(Wcor.sum())
        drop_P_se = se(Wdrop, P)
        drop_P_ae = float(np.abs(P[Wdrop] - G[Wdrop]).sum()) if Wdrop.any() else 0.0
        drop_n = int(Wdrop.sum())
        noise_P_se, noise_n = se(Wnoise, P), int(Wnoise.sum())
        keep_P_se, keep_n = se(Wkeep, P), int(Wkeep.sum())

        for g in groups:
            a = self.acc[g]
            a['cor_P_se'] += cor_P_se; a['cor_Sw_se'] += cor_Sw_se
            a['cor_n'] += cor_n; a['cor_sr_hit'] += sr_hit
            a['drop_P_se'] += drop_P_se; a['drop_P_ae'] += drop_P_ae; a['drop_n'] += drop_n
            a['noise_P_se'] += noise_P_se; a['noise_n'] += noise_n
            a['keep_P_se'] += keep_P_se; a['keep_n'] += keep_n

    @staticmethod
    def _rmse(se, n):
        return float(np.sqrt(se / n)) if n > 0 else float('nan')

    def reduce(self, group):
        a = self.acc[group]
        cor_P = self._rmse(a['cor_P_se'], a['cor_n'])
        cor_Sw = self._rmse(a['cor_Sw_se'], a['cor_n'])
        err = (1.0 - cor_P / cor_Sw) if (a['cor_n'] > 0 and cor_Sw > 0) else float('nan')
        sr = (a['cor_sr_hit'] / a['cor_n']) if a['cor_n'] > 0 else float('nan')
        dce = self._rmse(a['drop_P_se'], a['drop_n'])
        dce_mae = (a['drop_P_ae'] / a['drop_n']) if a['drop_n'] > 0 else float('nan')
        cpp = self._rmse(a['keep_P_se'], a['keep_n'])
        nrr = (self._rmse(a['noise_P_se'], a['noise_n']) / cpp) if (a['keep_n'] > 0 and cpp > 0) else float('nan')
        return dict(ERR=err, SR=sr, DCE=dce, DCE_mae=dce_mae, NRR=nrr, CPP=cpp,
                    n_cor=a['cor_n'], n_drop=a['drop_n'], n_noise=a['noise_n'], n_keep=a['keep_n'])

    def reduce_all(self):
        return {g: self.reduce(g) for g in self.acc}
