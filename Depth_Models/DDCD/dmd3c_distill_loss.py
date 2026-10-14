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

    # batch fallback
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
    Convert disp -> depth proxy using sparse GT calibration in inv-depth:
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


# -------------------------
# Energy-based reliability: E_S + E_A -> weight_map
# -------------------------
def _pool_sum(x, patch, stride, pad):
    return F.avg_pool2d(x, kernel_size=patch, stride=stride, padding=pad) * (patch * patch)


@torch.no_grad()
def compute_energy_weight_map(
    teacher_clean, teacher_noisy,
    gt_depth, gt_mask,
    patch=16, stride=1,
    beta=1.0,
    lambda_s=1.0, lambda_a=1.0,
    w_min=0.3,
    k_min_points=20,
    tau_mode="median",
    eps=1e-6,
):
    """
    Reliability weight map w(x) using:
      E_A: ambiguity energy from teacher_noisy vs teacher_clean (full image)
      E_S: sparse consistency energy from teacher_clean vs sparse GT (LiDAR points)
    Output:
      w_map: (B,1,H,W), in [w_min, 1]
      debug: dict with tau_s/tau_a and means
    """
    B, _, H, W = teacher_clean.shape
    m = gt_mask.float()
    pad = patch // 2

    tc = teacher_clean.clamp(min=eps)
    tn = teacher_noisy.clamp(min=eps)
    g = gt_depth.clamp(min=eps)

    # -------- E_A (ambiguity): patch mean of log-depth diff --------
    diff_a = (torch.log(tn) - torch.log(tc)).abs()  # (B,1,H,W)
    sum_a = _pool_sum(diff_a, patch, stride, pad)

    ones = torch.ones_like(diff_a)
    cnt_a = _pool_sum(ones, patch, stride, pad)
    mean_a = sum_a / (cnt_a + eps)

    # tau_A adaptive (global within batch)
    v_a = mean_a.reshape(-1)
    if v_a.numel() > 0:
        if tau_mode == "mean":
            tau_a = v_a.mean().clamp(min=eps)
        else:
            tau_a = v_a.median().clamp(min=eps)
    else:
        tau_a = teacher_clean.new_tensor(0.05)

    E_a = 1.0 - torch.exp(-mean_a / tau_a)

    # -------- E_S (sparse consistency): patch mean of log-depth error on sparse GT --------
    diff_s = m * (torch.log(tc) - torch.log(g)).abs()
    sum_s = _pool_sum(diff_s, patch, stride, pad)
    cnt_s = _pool_sum(m, patch, stride, pad)
    mean_s = sum_s / (cnt_s + eps)

    # density gate
    gS = torch.clamp(cnt_s / float(k_min_points), 0.0, 1.0)

    # tau_S adaptive on patches with points
    valid_s = (cnt_s > 0.5)
    if valid_s.any():
        v_s = mean_s[valid_s]
        if tau_mode == "mean":
            tau_s = v_s.mean().clamp(min=eps)
        else:
            tau_s = v_s.median().clamp(min=eps)
    else:
        tau_s = teacher_clean.new_tensor(0.05)

    E_s = 1.0 - torch.exp(-mean_s / tau_s)

    # -------- fuse energies -> weights --------
    E = (lambda_s * gS * E_s) + (lambda_a * E_a)
    w_patch = torch.exp(-beta * E)

    # if few points, avoid collapsing weights too much
    w_patch = gS * w_patch + (1.0 - gS) * w_min
    w_patch = w_patch.clamp(min=w_min, max=1.0)

    w_map = F.interpolate(w_patch, size=(H, W), mode="bilinear", align_corners=False)

    debug = {
        "tau_a": tau_a.detach(),
        "tau_s": tau_s.detach(),
        "Ea_mean": E_a.mean().detach(),
        "Es_mean": E_s.mean().detach(),
        "w_mean": w_map.mean().detach(),
    }
    return w_map, debug


class ScaleShiftInvariantLoss(nn.Module):
    """
    SSI distillation (L1 residual), with optional pixel-wise weight_map.
    """
    def __init__(self, deltas=None, eps=1e-6):
        super().__init__()
        self.deltas = deltas
        self.eps = eps

    def forward(self, outputs, teacher_depth, valid_mask=None, weight_map=None):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        deltas = [1.0] * len(outputs) if self.deltas is None else list(self.deltas)[:len(outputs)]
        losses = []

        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            wm = None
            if weight_map is not None:
                wm = F.interpolate(weight_map, size=pred.shape[-2:], mode="bilinear", align_corners=False).clamp(min=0.0)

            # align teacher -> pred (standard practice)
            s, b = solve_scale_shift(td, pred, mask, eps=self.eps)
            aligned = s * td + b

            m = mask.float()
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
    Multi-scale gradient regularization on residual R = pred - aligned(teacher),
    with optional weight_map.
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

    def forward(self, outputs, teacher_depth, valid_mask=None, weight_map=None):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        deltas = [1.0] * len(outputs) if self.deltas is None else list(self.deltas)[:len(outputs)]
        losses = []

        for pred, w in zip(outputs, deltas):
            td = F.interpolate(teacher_depth, size=pred.shape[-2:], mode="bilinear", align_corners=False)

            if valid_mask is None:
                mask = td > 0
            else:
                mask = _downsample_mask(valid_mask, pred.shape[-2:]) & (td > 0)

            wm0 = None
            if weight_map is not None:
                wm0 = F.interpolate(weight_map, size=pred.shape[-2:], mode="bilinear", align_corners=False).clamp(min=0.0)

            # align teacher
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
                    wk = None if wm0 is None else F.interpolate(wm0, size=Rk.shape[-2:], mode="bilinear", align_corners=False)

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
