import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def extract_anomaly_scores(outputs):
    """Extract anomaly probabilities from model outputs."""
    probs = torch.softmax(outputs, dim=1)
    if probs.shape[1] == 2:
        return probs[:, 1]  # Binary: direct anomaly probability
    else:
        return 1.0 - probs[:, 0]  # Multi-class: 1 - normal probability


def compute_auc_safe(y_true, y_scores):
    """Compute AUC with error handling."""
    try:
        # Convert to binary if needed
        binary_labels = (y_true > 0).astype(int) if y_true.max() > 1 else y_true
        return roc_auc_score(binary_labels, y_scores)
    except ValueError:
        # Only one class present
        return 0.5


def compute_youdens_j(fpr, tpr, thresholds):
    """Compute Youden's J statistic and best threshold."""
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    thresholds = np.asarray(thresholds)

    if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
        return float("nan"), float("nan")

    j_scores = tpr - fpr  # Youden's J
    best_idx = int(np.argmax(j_scores))

    return float(thresholds[best_idx]), float(j_scores[best_idx])
