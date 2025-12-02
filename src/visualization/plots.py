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


def plot_video_timeline(
    video_id, frame_indices, pred_scores, gt_mask, intervals, save_path
):
    """Plot anomaly score timeline for a single video."""
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot anomaly scores with seaborn color
    sns.lineplot(
        x=frame_indices,
        y=pred_scores,
        ax=ax,
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

    # ROC curve
    sns.lineplot(
        x=fpr,
        y=tpr,
        ax=ax,
        linewidth=2.5,
        label=f"ROC (AUC = {auc_score:.4f})",
        color=sns.color_palette("muted")[0],
    )

    # Random classifier baseline
    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        ax=ax,
        linewidth=2,
        linestyle="--",
        label="Random Classifier",
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


def plot_precision_recall_curve(precision, recall, save_path):
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.lineplot(
        x=recall,
        y=precision,
        ax=ax,
        linewidth=2.5,
        label="Precision-Recall",
        color=sns.color_palette("muted")[1],
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Frame-Level Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR curve: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
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
