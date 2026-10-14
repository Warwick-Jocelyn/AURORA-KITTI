"""
distill.py
==========
Self-contained distillation losses ported from:
  DMD3C/BP-Net/DMD3C/dmd3c_distill_loss.py  (Yongjie Wang, 2025)

Ported functions (verbatim, not reformulated):
  - disp_to_depth_proxy
  - _fit_inv_depth_affine_batch
  - solve_scale_shift
  - ScaleShiftInvariantLoss
  - MultiScaleResidualGradientLoss

Teacher normalisation replicates DMD3C/datasets/kitti.py  pull_DISP for da3:
  uint16 PNG -> depth_m = raw/256.0; clip(0,81); teacher_disp = depth_m/81*255

NO changes to DMD3C source files; NO changes to repo originals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Teacher loading / normalisation
# ---------------------------------------------------------------------------

def load_da3_as_disp(path: str) -> np.ndarray:
    """
    Load a da3 uint16 PNG and return teacher_disp in [0, 255] float32 (H, W).

    Replicates DMD3C/datasets/kitti.py  pull_DISP for da3:
        raw  = PIL.open(path)  [uint16, stored as int32 by PIL]
        if max(raw) > 255:  depth_m = raw / 256.0
        depth_m = clip(depth_m, 0, 81)
        teacher_disp = depth_m / 81.0 * 255.0
    """
    img = Image.open(path)
    raw = np.array(img, dtype=np.int32)
    if raw.max() > 255:
        depth_m = raw / 256.0
    else:
        depth_m = raw.astype(np.float32)
    depth_m = np.clip(depth_m, 0.0, 81.0)
    teacher_disp = (depth_m / 81.0 * 255.0).astype(np.float32)
    return teacher_disp  # (H, W)


# ---------------------------------------------------------------------------
# Low-level helpers (ported verbatim from dmd3c_distill_loss.py)
# ---------------------------------------------------------------------------

def _downsample_mask(mask: torch.Tensor, hw) -> torch.Tensor:
    if mask.dtype != torch.float32:
        mask = mask.float()
    m = F.interpolate(mask, size=hw, mode="nearest")
    return m > 0.5


@torch.no_grad()
def _fit_inv_depth_affine_batch(
    disp: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6,
    min_points: int = 50,
):
    """
    Fit per-image: 1/depth ≈ alpha * disp + beta  (least squares).
    Returns alpha, beta with shape (B, 1, 1, 1).
    Falls back to batch-level fit when too few valid points.
    """
    B = disp.shape[0]
    x = disp
    z = gt_depth.clamp(min=eps)
    y = 1.0 / z
    m = gt_mask.float()

    N = m.sum(dim=(2, 3), keepdim=True)          # (B,1,1,1)
    N_safe = N.clamp(min=1.0)

    x_mean = (m * x).sum(dim=(2, 3), keepdim=True) / N_safe
    y_mean = (m * y).sum(dim=(2, 3), keepdim=True) / N_safe

    xm = x - x_mean
    ym = y - y_mean

    var_x  = (m * xm * xm).sum(dim=(2, 3), keepdim=True) / N_safe
    cov_xy = (m * xm * ym).sum(dim=(2, 3), keepdim=True) / N_safe

    alpha = cov_xy / (var_x + eps)
    beta  = y_mean - alpha * x_mean

    # batch-level fallback
    Nb       = m.sum().clamp(min=1.0)
    xb_mean  = (m * x).sum() / Nb
    yb_mean  = (m * y).sum() / Nb
    var_xb   = (m * (x - xb_mean).pow(2)).sum() / Nb
    cov_b    = (m * (x - xb_mean) * (y - yb_mean)).sum() / Nb
    alpha_b  = cov_b / (var_xb + eps)
    beta_b   = yb_mean - alpha_b * xb_mean

    too_few = (N < float(min_points))
    if too_few.any():
        alpha = torch.where(too_few, alpha.new_full(alpha.shape, alpha_b.item()), alpha)
        beta  = torch.where(too_few, beta.new_full(beta.shape,  beta_b.item()),  beta)

    return alpha, beta


@torch.no_grad()
def disp_to_depth_proxy(
    disp: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6,
    min_points: int = 50,
) -> torch.Tensor:
    """
    Convert teacher disp -> depth proxy using sparse GT calibration in inv-depth:
        1/z ≈ alpha*disp + beta  =>  z_proxy = 1 / (alpha*disp + beta)

    All tensors: (B, 1, H, W), float32.
    """
    alpha, beta = _fit_inv_depth_affine_batch(
        disp, gt_depth, gt_mask, eps=eps, min_points=min_points
    )
    inv = alpha * disp + beta
    inv = inv.clamp(min=eps)
    return 1.0 / inv


def solve_scale_shift(
    dm: torch.Tensor,
    df: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Solve s, b minimising || df - (s*dm + b) ||_2 on mask, per-image.
    dm/df: (B,1,H,W); mask: bool (B,1,H,W).
    Returns s, b: (B,1,1,1).
    """
    m = mask.float()
    N = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

    mean_x = (m * dm).sum(dim=(2, 3), keepdim=True) / N
    mean_y = (m * df).sum(dim=(2, 3), keepdim=True) / N

    xm = dm - mean_x
    ym = df - mean_y

    var_x = (m * xm * xm).sum(dim=(2, 3), keepdim=True) / N
    cov   = (m * xm * ym).sum(dim=(2, 3), keepdim=True) / N

    s = cov / (var_x + eps)
    b = mean_y - s * mean_x
    return s, b


# ---------------------------------------------------------------------------
# Loss modules (ported verbatim from dmd3c_distill_loss.py)
# ---------------------------------------------------------------------------

class ScaleShiftInvariantLoss(nn.Module):
    """
    SSI distillation loss (L1 residual after affine alignment of teacher to pred).
    Accepts a single tensor (deltas=None → weight 1.0) or a list for multi-scale.
    """

    def __init__(self, deltas=None, eps: float = 1e-6):
        super().__init__()
        self.deltas = deltas
        self.eps = eps

    def forward(
        self,
        outputs,
        teacher_depth: torch.Tensor,
        valid_mask=None,
        weight_map=None,
    ):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        deltas = [1.0] * len(outputs) if self.deltas is None else list(self.deltas)[: len(outputs)]
        losses = []

        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            wm = None
            if weight_map is not None:
                wm = F.interpolate(
                    weight_map, size=pred.shape[-2:], mode="bilinear", align_corners=False
                ).clamp(min=0.0)

            s, b = solve_scale_shift(td, pred, mask, eps=self.eps)
            aligned = s * td + b

            m   = mask.float()
            res = (pred - aligned).abs()

            if wm is None:
                denom = m.sum().clamp(min=1.0)
                l = (m * res).sum() / denom
            else:
                w_eff = m * wm
                denom = w_eff.sum().clamp(min=1.0)
                l = (w_eff * res).sum() / denom

            losses.append(w * l)

        return losses


class MultiScaleResidualGradientLoss(nn.Module):
    """
    Multi-scale gradient regularisation on residual R = pred - aligned(teacher).
    Accepts a single tensor (deltas=None → weight 1.0).
    """

    def __init__(self, deltas=None, K: int = 4, eps: float = 1e-6):
        super().__init__()
        self.deltas = deltas
        self.K = K
        self.eps = eps

    @staticmethod
    def _grad_xy(t: torch.Tensor):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy

    def forward(
        self,
        outputs,
        teacher_depth: torch.Tensor,
        valid_mask=None,
        weight_map=None,
    ):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        deltas = [1.0] * len(outputs) if self.deltas is None else list(self.deltas)[: len(outputs)]
        losses = []

        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            wm0 = None
            if weight_map is not None:
                wm0 = F.interpolate(
                    weight_map, size=pred.shape[-2:], mode="bilinear", align_corners=False
                ).clamp(min=0.0)

            s, b = solve_scale_shift(td, pred, mask, eps=self.eps)
            aligned = s * td + b
            R = pred - aligned

            loss_k = 0.0
            for k in range(self.K):
                if k == 0:
                    Rk = R
                    wk = wm0
                else:
                    Rk = F.interpolate(R, scale_factor=1 / (2 ** k), mode="bilinear", align_corners=False)
                    wk = (
                        None
                        if wm0 is None
                        else F.interpolate(wm0, size=Rk.shape[-2:], mode="bilinear", align_corners=False)
                    )

                gx, gy = self._grad_xy(Rk)

                if wk is None:
                    loss_k = loss_k + gx.abs().mean() + gy.abs().mean()
                else:
                    wx = wk[..., :, 1:]
                    wy = wk[..., 1:, :]
                    denom_x = wx.mean().clamp(min=1e-6)
                    denom_y = wy.mean().clamp(min=1e-6)
                    loss_k = loss_k + (wx * gx.abs()).mean() / denom_x + (wy * gy.abs()).mean() / denom_y

            losses.append(w * (loss_k / float(self.K)))

        return losses
