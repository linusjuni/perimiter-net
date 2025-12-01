import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_video_timeline(
    video_id, frame_indices, pred_scores, gt_mask, intervals, save_path
):
    """
    Plot anomaly score timeline for a single video.

    Args:
        video_id: Video identifier
        frame_indices: List of frame indices
        pred_scores: Predicted anomaly scores
        gt_mask: Ground truth binary mask
        intervals: List of (start, end) tuples for ground truth
        save_path: Path to save plot
    """
    plt.figure(figsize=(15, 5))

    # Plot prediction scores
    plt.plot(frame_indices, pred_scores, "b-", linewidth=2, label="Anomaly Score")

    # Highlight ground truth regions
    for start, end in intervals:
        plt.axvspan(start, end, alpha=0.3, color="red", label="Ground Truth Anomaly")

    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)
    plt.title(f"{video_id}", fontsize=14, fontweight="bold")
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug(f"Saved timeline plot to {save_path}")


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """
    Plot ROC curve for frame-level evaluation.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Frame-Level ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_best_worst_videos(video_results, annotations, plot_dir, top_n=5):
    """
    Plot timelines for best and worst performing videos.

    Args:
        video_results: List of (video_id, auc, scores, labels, indices, intervals)
        annotations: Ground truth annotations dict
        plot_dir: Directory to save plots
        top_n: Number of best/worst videos to plot
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Sort by AUC
    video_results.sort(key=lambda x: x[1], reverse=True)

    # Top N best
    for i, (vid_id, auc, scores, labels, indices, intervals) in enumerate(
        video_results[:top_n]
    ):
        plot_video_timeline(
            f"{vid_id} (AUC={auc:.3f})",
            indices,
            scores,
            labels,
            intervals,
            save_path=plot_dir / f"best_{i + 1}_{vid_id}.png",
        )

    # Top N worst
    for i, (vid_id, auc, scores, labels, indices, intervals) in enumerate(
        video_results[-top_n:]
    ):
        plot_video_timeline(
            f"{vid_id} (AUC={auc:.3f})",
            indices,
            scores,
            labels,
            intervals,
            save_path=plot_dir / f"worst_{i + 1}_{vid_id}.png",
        )

    logger.info(f"Saved best/worst video plots to {plot_dir}")
