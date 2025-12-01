import torch
import csv
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingHistory:
    """Tracks and saves training/validation metrics to CSV."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / "training_history.csv"
        self.history = []

        # Initialize CSV with headers
        self.headers = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_auc",
            "val_anomaly_precision",
            "val_anomaly_recall",
            "val_anomaly_f1",
            "val_normal_precision",
            "val_normal_recall",
            "val_normal_f1",
            "val_tp",
            "val_fn",
            "val_tn",
            "val_fp",
            "val_fpr",
            "val_fnr",
            "learning_rate",
            "timestamp",
        ]

        # Create CSV if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created training history CSV at {self.csv_path}")

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_metrics,
        learning_rate: float,
    ):
        """
        Add new epoch results to history.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_metrics: EvaluationMetrics dataclass from validation
            learning_rate: Current learning rate
        """
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.acc,
            "val_auc": val_metrics.auc,
            "val_anomaly_precision": val_metrics.anomaly_precision,
            "val_anomaly_recall": val_metrics.anomaly_recall,
            "val_anomaly_f1": val_metrics.anomaly_f1,
            "val_normal_precision": val_metrics.normal_precision,
            "val_normal_recall": val_metrics.normal_recall,
            "val_normal_f1": val_metrics.normal_f1,
            "val_tp": val_metrics.tp,
            "val_fn": val_metrics.fn,
            "val_tn": val_metrics.tn,
            "val_fp": val_metrics.fp,
            "val_fpr": val_metrics.fpr,
            "val_fnr": val_metrics.fnr,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
        }

        self.history.append(row)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)

        logger.debug(f"Saved epoch {epoch} to training history")

    def get_best_epoch(self, metric: str = "val_auc") -> dict:
        """Return the epoch with the best value for the given metric."""
        if not self.history:
            return None

        if metric.startswith("val_loss"):
            best = min(self.history, key=lambda x: x[metric])
        else:
            best = max(self.history, key=lambda x: x[metric])

        return best


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth", is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    logger.debug(f"Saved checkpoint to {filepath}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """Load model checkpoint and optionally resume optimizer state."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)

    # Try to load best_auc
    best_auc = checkpoint.get("best_auc")

    logger.info(f"Resumed from epoch {epoch} with best AUC {best_auc:.4f}")
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience=7, min_delta=0.0, mode="max"):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.info(f"Early stopping initialized with patience={patience}, mode={mode}")

    def __call__(self, score):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(
                    f"Early stopping triggered after {self.patience} epochs without improvement"
                )

        return self.early_stop


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res if len(res) > 1 else res[0]
