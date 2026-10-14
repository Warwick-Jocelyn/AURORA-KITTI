import torch
import torch.nn as nn

class RandomProposalNormalizationLoss(nn.Module):
    def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1), num_proposals=32, min_crop_ratio=0.125, max_crop_ratio=0.5):
        super(RandomProposalNormalizationLoss, self).__init__()
        self.num_proposals = num_proposals
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.deltas = deltas

    def forward(self, outputs, target):
        """
        :param predicted_depth: Predicted depth map (B, 1, H, W)
        :param ground_truth_depth: Ground truth depth map (B, 1, H, W)
        :return: RPNL loss value
        """
        loss = [delta * self.compute_loss(ests, target) for ests, delta in zip(outputs, self.deltas)]
        return loss
        
    def compute_loss(self, predicted_depth, ground_truth_depth):
        B, _, H, W = predicted_depth.shape
        loss = 0.0

        for _ in range(self.num_proposals):
            # Randomly select crop size
            crop_ratio = torch.rand(1).item() * (self.max_crop_ratio - self.min_crop_ratio) + self.min_crop_ratio
            crop_h, crop_w = int(H * crop_ratio), int(W * crop_ratio)

            # Randomly select top-left corner of the crop
            top = torch.randint(0, H - crop_h + 1, (1,)).item()
            left = torch.randint(0, W - crop_w + 1, (1,)).item()

            # Extract patches
            pred_patch = predicted_depth[:, :, top:top + crop_h, left:left + crop_w]
            gt_patch = ground_truth_depth[:, :, top:top + crop_h, left:left + crop_w]

            # Flatten patches for normalization
            pred_patch = pred_patch.reshape(B, -1)
            gt_patch = gt_patch.reshape(B, -1)

            # Normalize patches using median absolute deviation normalization
            pred_median = pred_patch.median(dim=1, keepdim=True)[0]
            gt_median = gt_patch.median(dim=1, keepdim=True)[0]

            pred_mad = torch.median(torch.abs(pred_patch - pred_median), dim=1, keepdim=True)[0]
            gt_mad = torch.median(torch.abs(gt_patch - gt_median), dim=1, keepdim=True)[0]

            pred_patch_norm = (pred_patch - pred_median) / (pred_mad + 1e-6)
            gt_patch_norm = (gt_patch - gt_median) / (gt_mad + 1e-6)

            # Calculate the L1 difference between normalized patches
            patch_loss = torch.mean(torch.abs(pred_patch_norm - gt_patch_norm))
            loss += patch_loss

        loss /= self.num_proposals
        return loss

# Example usage:
if __name__ == "__main__":
    # Create random predicted and ground truth depth maps
    predicted_depth = torch.rand(2, 1, 256, 256)  # (batch_size, channels, height, width)
    ground_truth_depth = torch.rand(2, 1, 256, 256)

    # Initialize the loss function
    rpn_loss = RandomProposalNormalizationLoss()

    # Compute the loss
    loss_value = rpn_loss(predicted_depth, ground_truth_depth)
    print(f"RPN Loss: {loss_value.item()}")

class RegularizationLoss(nn.Module):
    """
    Enforce losses on pixels without any gts.
    """
    def __init__(self, loss_weight=0.1):
        super(RegularizationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6

    def forward(self, prediction, gt):
        mask = gt > 1e-3
        pred_wo_gt = prediction[~mask]
        loss = 1/ (torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + self.eps))
        return loss * self.loss_weight

class GradientMatchingLoss(nn.Module):
    """
    梯度蒸馏 loss: 约束 pred 与 teacher 的 depth 梯度一致,
    使边缘更锐利, 与深度不连续对齐.
    """
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, preds, teacher, mask=None):
        if isinstance(preds, (list, tuple)):
            losses = []
            for p in preds:
                t = F.interpolate(teacher, size=p.shape[-2:], mode='bilinear', align_corners=False)
                m = None
                if mask is not None:
                    m = F.interpolate(mask.float(), size=p.shape[-2:], mode='nearest') > 0.5
                losses.append(self.compute_single(p, t, m))
            return [self.loss_weight * l for l in losses]
        else:
            t = F.interpolate(teacher, size=preds.shape[-2:], mode='bilinear', align_corners=False)
            m = None
            if mask is not None:
                m = F.interpolate(mask.float(), size=preds.shape[-2:], mode='nearest') > 0.5
            return self.loss_weight * self.compute_single(preds, t, m)

    def compute_single(self, pred, teacher, mask):
        # pred, teacher: (B,1,H,W)
        def gradient_x(img):
            return img[:, :, :, 1:] - img[:, :, :, :-1]

        def gradient_y(img):
            return img[:, :, 1:, :] - img[:, :, :-1, :]

        pred_gx = gradient_x(pred)
        pred_gy = gradient_y(pred)
        teach_gx = gradient_x(teacher)
        teach_gy = gradient_y(teacher)

        if mask is not None:
            # 对 mask 做同样的裁切
            mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
            gx_diff = (pred_gx - teach_gx)[mask_x]
            gy_diff = (pred_gy - teach_gy)[mask_y]
        else:
            gx_diff = pred_gx - teach_gx
            gy_diff = pred_gy - teach_gy

        loss = gx_diff.abs().mean() + gy_diff.abs().mean()
        return loss
