import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MILRankingLoss(nn.Module):
    """Deep MIL ranking loss."""

    def __init__(self, lambda_1=1e-5, lambda_2=1e-5):
        super(MILRankingLoss, self).__init__()
        self.lambda_1 = lambda_1  # Sparsity
        self.lambda_2 = lambda_2  # Smoothness

    def forward(self, preds_normal, preds_anomaly):
        """Compute MIL ranking loss."""
        preds_normal = preds_normal.squeeze(-1)
        preds_anomaly = preds_anomaly.squeeze(-1)
        max_normal = torch.max(preds_normal, dim=1)[0]
        max_anomaly = torch.max(preds_anomaly, dim=1)[0]
        loss_rank = torch.mean(torch.clamp(1.0 - max_anomaly + max_normal, min=0))
        loss_sparsity = torch.mean(torch.sum(preds_anomaly, dim=1)) * self.lambda_1
        diff = preds_anomaly[:, 1:] - preds_anomaly[:, :-1]
        loss_smoothness = torch.mean(torch.sum(diff**2, dim=1)) * self.lambda_2
        total_loss = loss_rank + loss_sparsity + loss_smoothness
        return total_loss, {
            "rank": loss_rank.item(),
            "sparse": loss_sparsity.item(),
            "smooth": loss_smoothness.item(),
        }
