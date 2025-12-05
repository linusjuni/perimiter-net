import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

# Ensure repo root is on path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def parse_args():
    default_root = Path(__file__).parent.parent.parent / "results" / "late_fusion"
    parser = argparse.ArgumentParser(
        description="Plot ROC/PR/confusion matrix for late-fusion MIL results."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=False,
        help="Path to a late fusion run directory containing metrics.json and frame_scores_labels.npz.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory to search for runs if --run-dir is not provided.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold to recompute confusion matrix (defaults to metrics.json threshold if omitted).",
    )
    return parser.parse_args()


def find_latest_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under {root}")
    return run_dirs[-1]


def main():
    args = parse_args()
    run_dir = args.run_dir or find_latest_run(args.root)

    metrics_path = run_dir / "metrics.json"
    arrays_path = run_dir / "frame_scores_labels.npz"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    if not arrays_path.exists():
        raise FileNotFoundError(f"frame_scores_labels.npz not found at {arrays_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    npz = np.load(arrays_path)
    y_scores = npz["scores"]
    y_true = npz["labels"]

    # ROC / PR
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    # Confusion matrix
    thr = args.threshold
    if thr is None:
        thr = metrics["metrics"].get("best_threshold", 0.5)
    y_pred = (y_scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Output dir for plots
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(fpr, tpr, roc_auc, plot_dir / "roc_curve.png")
    plot_precision_recall_curve(
        precision,
        recall,
        plot_dir / "pr_curve.png",
        positive_rate=float(y_true.mean()),
    )
    plot_confusion_matrix(
        cm,
        class_names=["Normal", "Anomaly"],
        save_path=plot_dir / "confusion_matrix.png",
        normalize=False,
        threshold=thr,
    )
    plot_confusion_matrix(
        cm,
        class_names=["Normal", "Anomaly"],
        save_path=plot_dir / "confusion_matrix_norm.png",
        normalize=True,
        threshold=thr,
    )

    print(f"Saved plots to {plot_dir}")
    print(f"ROC AUC: {roc_auc:.4f} | Threshold used for CM: {thr:.4f}")
    print(f"ROC thresholds (sample): min {roc_thresholds.min():.4f}, max {roc_thresholds.max():.4f}")
    print(f"PR thresholds count: {len(pr_thresholds)}")


if __name__ == "__main__":
    main()
