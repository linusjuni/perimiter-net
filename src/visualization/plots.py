import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("muted")


def _downsample_xy(x, y, max_points=5000):
    """Downsample paired arrays for faster plotting without changing the curve shape."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def plot_video_timeline(
    video_id, frame_indices, pred_scores, gt_mask, intervals, save_path
):
    """Plot anomaly score timeline for a single video."""
    fig, ax = plt.subplots(figsize=(15, 5))

    x, y = _downsample_xy(frame_indices, pred_scores, max_points=8000)

    # Plot anomaly scores with seaborn color
    ax.plot(
        x,
        y,
        linewidth=2,
        label="Anomaly Score",
        color=sns.color_palette("muted")[0],
    )

    # Highlight anomaly regions
    for start, end in intervals:
        ax.axvspan(
            start,
            end,
            alpha=0.25,
            color=sns.color_palette("muted")[3],
            label="Ground Truth Anomaly",
        )

    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("Anomaly Score", fontsize=12)
    ax.set_title(f"{video_id}", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug(f"Saved timeline: {save_path.name}")


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 8))

    x, y = _downsample_xy(fpr, tpr, max_points=20000)

    # ROC curve
    ax.plot(
        x,
        y,
        linewidth=2.5,
        label=f"ROC (AUC = {auc_score:.4f})",
        color=sns.color_palette("muted")[0],
    )

    # Random classifier baseline
    ax.plot(
        [0, 1],
        [0, 1],
        linewidth=2,
        linestyle="--",
        label="No-skill classifier",
        color=sns.color_palette("muted")[3],
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Frame-Level ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curve: {save_path}")


def plot_precision_recall_curve(precision, recall, save_path, positive_rate=None):
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 8))

    x, y = _downsample_xy(recall, precision, max_points=20000)

    ax.plot(
        x,
        y,
        linewidth=2.5,
        label="Precision-Recall",
        color=sns.color_palette("muted")[1],
    )

    if positive_rate is not None:
        ax.axhline(
            positive_rate,
            linewidth=2,
            linestyle="--",
            color=sns.color_palette("muted")[3],
            label=f"No-skill (p={positive_rate:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Frame-Level Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR curve: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, normalize=False, threshold=None):
    """Plot confusion matrix as heatmap."""
    cm = np.array(cm)
    display_cm = cm.copy()

    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(
            display_cm, row_sums, out=np.zeros_like(display_cm, dtype=float), where=row_sums != 0
        )
        fmt = ".2f"
    else:
        display_cm = display_cm.astype(np.int64)
        fmt = "d"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        display_cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    if threshold is not None:
        title = f"{title} (thr={threshold:.3f})"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def plot_best_worst_videos(video_results, annotations, plot_dir, top_n=5):
    """Plot timelines for best and worst performing videos."""
    plot_dir = Path(plot_dir)

    video_results.sort(key=lambda x: x[1], reverse=True)

    # Best
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

    # Worst
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

    logger.info(f"Saved {top_n * 2} video timelines")
