import time
import json
import torch
import numpy as np
from dataclasses import asdict, dataclass
from src.utils.evaluation_utils import compute_auc_safe
from src.utils.training_utils import AverageMeter
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MILMetrics:
    """
    Structured container for MIL evaluation results.
    Stores video-level AUC and loss components.
    """

    loss: float
    rank_loss: float
    sparsity_loss: float
    smoothness_loss: float
    auc: float

    def to_dict(self):
        """Convert to dictionary for JSON serialization or logging."""
        return asdict(self)

    def save_to_json(self, path):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self):
        """Pretty print for logging."""
        return (
            f"Loss: {self.loss:.4f} (Rank: {self.rank_loss:.4f}, "
            f"Sparse: {self.sparsity_loss:.6f}, Smooth: {self.smoothness_loss:.6f}) | "
            f"AUC: {self.auc:.4f}"
        )


def evaluate_mil(
    model: torch.nn.Module,
    loader,  # MILDataLoader
    criterion,  # MILRankingLoss
    device: torch.device,
    split: str = "val",
) -> MILMetrics:
    """
    Evaluate MIL model on video-level anomaly detection.

    Computes:
    - Video-level AUC (max score per video)
    - MIL loss components (ranking, sparsity, smoothness)

    Args:
        model: MIL model
        loader: MILDataLoader instance
        criterion: MILRankingLoss instance
        device: Device to evaluate on
        split: Split name for logging ('val' or 'test')

    Returns:
        MILMetrics dataclass
    """
    model.eval()
    start_time = time.time()

    # Meters for loss components
    loss_meter = AverageMeter()
    rank_meter = AverageMeter()
    sparsity_meter = AverageMeter()
    smoothness_meter = AverageMeter()

    # Lists to store video-level scores
    all_normal_scores = []
    all_anomaly_scores = []

    with torch.no_grad():
        # Process Normal videos
        for features in loader.normal_videos:
            features = (
                torch.from_numpy(features).float().unsqueeze(0).to(device)
            )  # (1, 32, 512)
            scores = model(features)  # (1, 32, 1)
            max_score = torch.max(scores).item()
            all_normal_scores.append(max_score)

        # Process Anomaly videos
        for features in loader.anomaly_videos:
            features = (
                torch.from_numpy(features).float().unsqueeze(0).to(device)
            )  # (1, 32, 512)
            scores = model(features)  # (1, 32, 1)
            max_score = torch.max(scores).item()
            all_anomaly_scores.append(max_score)

        # Compute loss on a sample batch (for monitoring)
        # We need balanced batches for the ranking loss
        batch_size = min(30, len(loader.normal_videos), len(loader.anomaly_videos))
        norm_batch, anom_batch = loader.get_batch(batch_size)
        norm_batch = norm_batch.to(device)
        anom_batch = anom_batch.to(device)

        norm_preds = model(norm_batch)
        anom_preds = model(anom_batch)

        total_loss, loss_components = criterion(norm_preds, anom_preds)

        loss_meter.update(total_loss.item(), batch_size)
        rank_meter.update(loss_components["rank"], batch_size)
        sparsity_meter.update(loss_components["sparse"], batch_size)
        smoothness_meter.update(loss_components["smooth"], batch_size)

    # Compute video-level AUC
    y_true = np.concatenate(
        [
            np.zeros(len(all_normal_scores)),  # Normal = 0
            np.ones(len(all_anomaly_scores)),  # Anomaly = 1
        ]
    )
    y_scores = np.concatenate([all_normal_scores, all_anomaly_scores])

    auc = compute_auc_safe(y_true, y_scores)

    duration = time.time() - start_time

    metrics = MILMetrics(
        loss=loss_meter.avg,
        rank_loss=rank_meter.avg,
        sparsity_loss=sparsity_meter.avg,
        smoothness_loss=smoothness_meter.avg,
        auc=auc,
    )

    logger.info(f"[{split.upper()}] Evaluation [{duration:.0f}s] - {metrics}")

    return metrics


@dataclass
class TwoStreamMILMetrics:
    """
    Structured container for Two-Stream MIL evaluation results.
    """

    loss: float
    auc: float
    num_normal: int
    num_anomaly: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization or logging."""
        return asdict(self)

    def save_to_json(self, path):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self):
        """Pretty print for logging."""
        return (
            f"Loss: {self.loss:.4f} | AUC: {self.auc:.4f} | "
            f"Videos: {self.num_normal} normal, {self.num_anomaly} anomaly"
        )


def evaluate_two_stream(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    split: str = "val",
) -> TwoStreamMILMetrics:
    """Evaluate two-stream MIL model."""
    model.eval()
    start_time = time.time()

    all_scores = []
    all_labels = []
    losses = AverageMeter()

    with torch.no_grad():
        # Evaluate all normal videos
        for video in loader.normal_videos:
            rgb = torch.from_numpy(video["rgb"]).float().unsqueeze(0).to(device)
            motion = torch.from_numpy(video["motion"]).float().unsqueeze(0).to(device)

            scores = model(rgb, motion)  # (1, 32, 1)
            max_score = scores.max().item()

            all_scores.append(max_score)
            all_labels.append(0)

        # Evaluate all anomaly videos
        for video in loader.anomaly_videos:
            rgb = torch.from_numpy(video["rgb"]).float().unsqueeze(0).to(device)
            motion = torch.from_numpy(video["motion"]).float().unsqueeze(0).to(device)

            scores = model(rgb, motion)  # (1, 32, 1)
            max_score = scores.max().item()

            all_scores.append(max_score)
            all_labels.append(1)

    # Compute AUC
    auc = compute_auc_safe(all_labels, all_scores)

    # Compute loss on a few batches for monitoring
    num_eval_batches = min(5, loader.get_num_batches(30))
    for _ in range(num_eval_batches):
        norm_rgb, norm_motion, anom_rgb, anom_motion = loader.get_batch(30)

        norm_rgb = norm_rgb.to(device)
        norm_motion = norm_motion.to(device)
        anom_rgb = anom_rgb.to(device)
        anom_motion = anom_motion.to(device)

        preds_normal = model(norm_rgb, norm_motion)
        preds_anomaly = model(anom_rgb, anom_motion)

        loss, _ = criterion(preds_normal, preds_anomaly)
        losses.update(loss.item(), 30)

    duration = time.time() - start_time

    metrics = TwoStreamMILMetrics(
        loss=losses.avg,
        auc=auc,
        num_normal=len(loader.normal_videos),
        num_anomaly=len(loader.anomaly_videos),
    )

    logger.info(f"[{split.upper()}] Evaluation [{duration:.0f}s] - {metrics}")

    return metrics
