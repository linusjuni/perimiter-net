import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def extract_anomaly_scores(outputs):
    """
    Extract anomaly probabilities from model outputs.
    Assumes Class 0 is ALWAYS 'Normal', Class 1+ is 'Anomaly'.

    Args:
        outputs: Model logits (N, num_classes)

    Returns:
        torch.Tensor: Anomaly scores (N,) in range [0, 1]
    """
    probs = torch.softmax(outputs, dim=1)
    if probs.shape[1] == 2:
        return probs[:, 1]  # Binary: direct anomaly probability
    else:
        return 1.0 - probs[:, 0]  # Multi-class: 1 - normal probability


def compute_auc_safe(y_true, y_scores):
    """
    Compute AUC with error handling for edge cases.

    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted scores

    Returns:
        float: AUC score or 0.5 if computation fails
    """
    try:
        # Convert to binary if needed
        binary_labels = (y_true > 0).astype(int) if y_true.max() > 1 else y_true
        return roc_auc_score(binary_labels, y_scores)
    except ValueError:
        # Only one class present
        return 0.5


def compute_youdens_j(fpr, tpr, thresholds):
    """
    Compute Youden's J statistic (TPR - FPR) and return the best threshold.

    Args:
        fpr (array-like): False positive rates from roc_curve.
        tpr (array-like): True positive rates from roc_curve.
        thresholds (array-like): Thresholds corresponding to fpr/tpr.

    Returns:
        tuple: (best_threshold, best_j_score)
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    thresholds = np.asarray(thresholds)

    if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
        return float("nan"), float("nan")

    j_scores = tpr - fpr  # Youden's J
    best_idx = int(np.argmax(j_scores))

    return float(thresholds[best_idx]), float(j_scores[best_idx])
