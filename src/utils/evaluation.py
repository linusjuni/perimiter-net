import time
import json
import torch
import numpy as np
from dataclasses import asdict, dataclass
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.amp import autocast
from src.utils.training_utils import AverageMeter, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Structured container for evaluation results.
    Stores RAW counts for binary tasks to allow deriving any metric later.
    """

    loss: float
    acc: float
    auc: float

    # Per-Class Metrics
    anomaly_precision: float
    anomaly_recall: float
    anomaly_f1: float

    normal_precision: float
    normal_recall: float
    normal_f1: float

    # Raw Confusion Matrix Counts (Integers)
    # Default to 0 for multi-class where 2x2 matrix doesn't apply directly
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Computed Properties
    @property
    def fpr(self) -> float:
        """False Positive Rate (False Alarm Rate) = FP / (FP + TN)"""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate (Missed Crime Rate) = FN / (FN + TP)"""
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization or logging."""
        return asdict(self)

    def save_to_json(self, path):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self):
        """Pretty print with derived rates."""
        return (
            f"Loss: {self.loss:.4f} | Acc: {self.acc:.2f}% | AUC: {self.auc:.4f}\n"
            f"   >> Anomaly: P={self.anomaly_precision:.3f} R={self.anomaly_recall:.3f} F1={self.anomaly_f1:.3f}\n"
            f"   >> Normal:  P={self.normal_precision:.3f} R={self.normal_recall:.3f}\n"
            f"   >> Counts:  TP={self.tp} FN={self.fn} (Missed) | TN={self.tn} FP={self.fp} (False Alarm)\n"
            f"   >> Rates:   FPR={self.fpr:.4f} | FNR={self.fnr:.4f}"
        )


def _extract_anomaly_scores(outputs):
    """
    Extracts anomaly probabilities (score 0.0 to 1.0).
    Assumes Class 0 is ALWAYS 'Normal'.
    """
    probs = torch.softmax(outputs, dim=1)
    if probs.shape[1] == 2:
        return probs[:, 1]  # Binary
    else:
        return 1.0 - probs[:, 0]  # Multi-class


def compute_metrics_dataclass(
    y_true, y_pred, y_probs, avg_loss, avg_acc, num_classes
) -> EvaluationMetrics:
    # 1. AUC
    try:
        binary_labels = (y_true > 0).astype(int)
        auc_score = roc_auc_score(binary_labels, y_probs)
    except ValueError:
        auc_score = 0.5

    # 2. Precision/Recall/F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Safe indexing
    norm_p = prec[0] if len(prec) > 0 else 0.0
    norm_r = rec[0] if len(rec) > 0 else 0.0
    norm_f1 = f1[0] if len(f1) > 0 else 0.0

    anom_p = prec[1] if len(prec) > 1 else 0.0
    anom_r = rec[1] if len(rec) > 1 else 0.0
    anom_f1 = f1[1] if len(f1) > 1 else 0.0

    # 3. Raw Counts (Binary Only)
    tp, tn, fp, fn = 0, 0, 0, 0
    if num_classes == 2:
        try:
            # Note: scikit-learn confusion_matrix returns [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            pass

    return EvaluationMetrics(
        loss=avg_loss,
        acc=avg_acc,
        auc=auc_score,
        anomaly_precision=float(anom_p),
        anomaly_recall=float(anom_r),
        anomaly_f1=float(anom_f1),
        normal_precision=float(norm_p),
        normal_recall=float(norm_r),
        normal_f1=float(norm_f1),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    split: str = "val",
) -> EvaluationMetrics:
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    # Pre-allocate
    num_samples = len(data_loader.dataset)
    all_probs = np.zeros(num_samples, dtype=np.float32)
    all_labels = np.zeros(num_samples, dtype=np.int32)
    all_preds = np.zeros(num_samples, dtype=np.int32)

    ptr = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast("cuda", enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            acc = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc.item(), inputs.size(0))

            probs_scores = _extract_anomaly_scores(outputs)
            _, preds = torch.max(outputs, 1)

            # Fill arrays
            batch_size = inputs.size(0)
            end_ptr = min(ptr + batch_size, num_samples)
            count = end_ptr - ptr

            if count > 0:
                all_probs[ptr:end_ptr] = probs_scores.cpu().numpy()[:count]
                all_labels[ptr:end_ptr] = targets.cpu().numpy()[:count]
                all_preds[ptr:end_ptr] = preds.cpu().numpy()[:count]
                ptr += count

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"[{split.upper()}] Batch {batch_idx + 1}/{len(data_loader)} Loss: {losses.avg:.4f}"
                )

    # Trim
    all_probs = all_probs[:ptr]
    all_labels = all_labels[:ptr]
    all_preds = all_preds[:ptr]

    # Compute Metrics
    num_classes = model.fc.out_features if hasattr(model, "fc") else 2
    metrics = compute_metrics_dataclass(
        all_labels, all_preds, all_probs, losses.avg, accs.avg, num_classes
    )

    duration = time.time() - start_time
    logger.info(f"[{split.upper()}] Finished in {duration:.0f}s")
    logger.info(f"Results:\n{metrics}")

    return metrics
