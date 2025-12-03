import sys
import pickle
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
        print("Usage: python plot_r3d_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Generating plots for: {results_dir}")

    # Create plots subdirectory
    #plot_dir = results_dir / "plots" # choose this
    plot_dir = Path("plots") # or this
    plot_dir.mkdir(exist_ok=True)

    # Load raw data
    data = np.load(results_dir / "raw_data.npz")
    fpr = data["fpr"]
    tpr = data["tpr"]
    frame_auc = float(data["frame_auc"])
    precision = data["precision"]
    recall = data["recall"]
    confusion = data["confusion"]
    decision_threshold = (
        float(data["decision_threshold"]) if "decision_threshold" in data.files else None
    )
    positive_rate = float(np.mean(data["labels"])) if "labels" in data.files else None
    run_label = str(data["run_name"]) if "run_name" in data.files else results_dir.name

    # Load video results
    with open(results_dir / "video_results.pkl", "rb") as f:
        video_results = pickle.load(f)

    # Generate ROC curve
    logger.info("Plotting ROC curve...")
    plot_roc_curve(fpr, tpr, frame_auc, plot_dir / f"roc_curve_{run_label}.png")

    # Generate Precision-Recall curve
    logger.info("Plotting Precision-Recall curve...")
    plot_precision_recall_curve(
        precision,
        recall,
        plot_dir / f"precision_recall_{run_label}.png",
        positive_rate=positive_rate,
    )

    # Generate confusion matrix
    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(
        confusion,
        ["Normal", "Anomaly"],
        plot_dir / f"confusion_matrix_{run_label}.png",
        threshold=decision_threshold,
    )

    # Generate best/worst videos
    if video_results:
        logger.info("Plotting best/worst videos...")
        plot_best_worst_videos(video_results, {}, plot_dir, top_n=5)

    logger.info(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
