import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_best_worst_videos,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_mil_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Generating MIL plots for: {results_dir}")

    plot_dir = Path("plots") / results_dir.name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    data = np.load(results_dir / "raw_data.npz", allow_pickle=True)
    fpr = data["fpr"]
    tpr = data["tpr"]
    segment_auc = float(data["segment_auc"])
    precision = data["precision"]
    recall = data["recall"]
    confusion = data["confusion"]
    decision_threshold = (
        float(data["decision_threshold"]) if "decision_threshold" in data.files else None
    )
    positive_rate = float(np.mean(data["labels"])) if "labels" in data.files else None
    run_label = str(data["run_name"]) if "run_name" in data.files else results_dir.name

    # Load video results (optional)
    video_results_path = results_dir / "video_results.npy"
    video_results = []
    if video_results_path.exists():
        vr = np.load(video_results_path, allow_pickle=True)
        # Each entry: (video_id, video_auc, seg_scores, seg_labels, segment_frame_centers, intervals)
        for item in vr:
            video_id, video_auc, seg_scores, seg_labels, frame_centers, intervals = item
            video_results.append(
                (video_id, video_auc, seg_scores, seg_labels, frame_centers, intervals)
            )
    else:
        logger.warning("video_results.npy not found; skipping best/worst plots.")

    # ROC curve
    logger.info("Plotting ROC curve...")
    plot_roc_curve(
        fpr, tpr, segment_auc, plot_dir / f"roc_curve_{run_label}.png", title="Segment-Level ROC Curve"
    )

    # Precision-Recall curve
    logger.info("Plotting Precision-Recall curve...")
        plot_precision_recall_curve(
            precision,
            recall,
            plot_dir / f"precision_recall_{run_label}.png",
            positive_rate=positive_rate,
            title="Segment-Level Precision-Recall Curve",
        )

    # Confusion matrix
    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(
        confusion,
        ["Normal", "Anomaly"],
        plot_dir / f"confusion_matrix_{run_label}.png",
        threshold=decision_threshold,
    )

    # Best/Worst videos (by segment-level AUC)
    if video_results:
        # Filter out NaN AUCs
        video_results = [vr for vr in video_results if not np.isnan(vr[1])]
        if video_results:
            logger.info("Plotting best/worst videos...")
            plot_best_worst_videos(
                video_results, {}, plot_dir, top_n=5, x_label="Frame Index"
            )
        else:
            logger.info("All video AUCs are NaN; skipping best/worst plots.")

    logger.info(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
