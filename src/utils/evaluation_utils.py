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
    except ValueError as e:
        # Only one class present
        return 0.5
