# dmd3c_distill_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _downsample_mask(mask, hw):
    if mask.dtype != torch.float32:
        mask = mask.float()
    m = F.interpolate(mask, size=hw, mode="nearest")
    return m > 0.5

@torch.no_grad()
def _fit_inv_depth_affine_batch(disp, gt_depth, gt_mask, eps=1e-6, min_points=50):
    """
    Fit per-image: 1/depth ≈ alpha * disp + beta  (least squares)
    Returns alpha,beta with shape (B,1,1,1).
    If too few valid points, fallback to batch-level fit.
    """
    B = disp.shape[0]
    x = disp
    z = gt_depth.clamp(min=eps)
    y = 1.0 / z
    m = gt_mask.float()

    # per-image sums
    N = m.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)
    N_safe = N.clamp(min=1.0)

    x_mean = (m * x).sum(dim=(2, 3), keepdim=True) / N_safe
    y_mean = (m * y).sum(dim=(2, 3), keepdim=True) / N_safe

    xm = x - x_mean
    ym = y - y_mean

    var_x = (m * xm * xm).sum(dim=(2, 3), keepdim=True) / N_safe
    cov_xy = (m * xm * ym).sum(dim=(2, 3), keepdim=True) / N_safe

    alpha = cov_xy / (var_x + eps)
    beta = y_mean - alpha * x_mean

    # batch-level fallback (aggregate all valid points in batch)
    Nb = m.sum().clamp(min=1.0)
    xb_mean = (m * x).sum() / Nb
    yb_mean = (m * y).sum() / Nb
    var_xb = (m * (x - xb_mean).pow(2)).sum() / Nb
    cov_b = (m * (x - xb_mean) * (y - yb_mean)).sum() / Nb

    alpha_b = cov_b / (var_xb + eps)
    beta_b = yb_mean - alpha_b * xb_mean

    too_few = (N < float(min_points))
    if too_few.any():
        alpha = torch.where(too_few, alpha.new_full(alpha.shape, alpha_b), alpha)
        beta = torch.where(too_few, beta.new_full(beta.shape, beta_b), beta)

    return alpha, beta

@torch.no_grad()
def disp_to_depth_proxy(disp, gt_depth, gt_mask, eps=1e-6, min_points=50):
    """
    Convert DA disparity/relative disp -> depth proxy with metric-consistent scale
    using sparse GT calibration in inv-depth space:
        1/z ≈ alpha*disp + beta  =>  z_proxy = 1/(alpha*disp + beta)
    """
    alpha, beta = _fit_inv_depth_affine_batch(disp, gt_depth, gt_mask, eps=eps, min_points=min_points)
    inv = alpha * disp + beta
    inv = inv.clamp(min=eps)
    depth_proxy = 1.0 / inv
    return depth_proxy

def solve_scale_shift(dm, df, mask, eps=1e-6):
    """
    Solve s,b minimizing || df - (s*dm + b) ||_2 on mask, per-image.
    dm/df: (B,1,H,W), mask: bool (B,1,H,W)
    return s,b: (B,1,1,1)
    """
    m = mask.float()
    N = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

    mean_x = (m * dm).sum(dim=(2, 3), keepdim=True) / N
    mean_y = (m * df).sum(dim=(2, 3), keepdim=True) / N

    xm = dm - mean_x
    ym = df - mean_y

    var_x = (m * xm * xm).sum(dim=(2, 3), keepdim=True) / N
    cov = (m * xm * ym).sum(dim=(2, 3), keepdim=True) / N

    s = cov / (var_x + eps)
    b = mean_y - s * mean_x
    return s, b

class ScaleShiftInvariantLoss(nn.Module):
    """
    True SSI distillation: min_{s,b} |Df - (s*Dm + b)| (L1 residual),
    where (s,b) solved via L2 on valid pixels (standard practice).
    """
    def __init__(self, deltas=None, eps=1e-6):
        super().__init__()
        self.deltas = deltas
        self.eps = eps

    def forward(self, outputs, teacher_depth, valid_mask=None):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if self.deltas is None:
            deltas = [1.0] * len(outputs)
        else:
            deltas = list(self.deltas)[:len(outputs)]

        losses = []
        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            s, b = solve_scale_shift(td, pred, mask, eps=self.eps)
            aligned = s * td + b

            m = mask.float()
            denom = m.sum().clamp(min=1.0)
            l = (m * (pred - aligned).abs()).sum() / denom
            losses.append(w * l)

        return losses

class MultiScaleResidualGradientLoss(nn.Module):
    """
    Paper-style multi-scale gradient regularization on residual R = Df - Dm (aligned),
    across K pyramid levels.
    """
    def __init__(self, deltas=None, K=4, eps=1e-6):
        super().__init__()
        self.deltas = deltas
        self.K = K
        self.eps = eps

    @staticmethod
    def _grad_xy(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy

    def forward(self, outputs, teacher_depth, valid_mask=None):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if self.deltas is None:
            deltas = [1.0] * len(outputs)
        else:
            deltas = list(self.deltas)[:len(outputs)]

        losses = []
        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            # align teacher first (stabilizes residual)
            s, b = solve_scale_shift(td, pred, mask, eps=self.eps)
            aligned = s * td + b

            R = pred - aligned

            loss_k = 0.0
            for k in range(self.K):
                if k == 0:
                    Rk = R
                else:
                    Rk = F.interpolate(R, scale_factor=1 / (2 ** k), mode="bilinear", align_corners=False)

                gx, gy = self._grad_xy(Rk)
                loss_k = loss_k + gx.abs().mean() + gy.abs().mean()

            losses.append(w * (loss_k / float(self.K)))

        return losses
