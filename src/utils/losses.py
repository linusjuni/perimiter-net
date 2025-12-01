import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification."""

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
        """
        Args:
            inputs (torch.Tensor): Logits from model, shape (N, C)
            targets (torch.Tensor): Ground truth class indices, shape (N,)

        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get the probability of the true class (p_t)
        p_t = torch.exp(-ce_loss)

        # Compute focal loss components
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        # Apply alpha (class balancing) if provided
        if self.alpha is not None:
            # Gather alpha values for the specific targets in the batch
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
