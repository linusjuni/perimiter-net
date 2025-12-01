import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.r3d import create_r3d_classifier
from src.datasets.transforms import RGBVideoTransform
from src.utils.frame_level_evaluation import evaluate_frame_level
from src.utils.training_utils import load_checkpoint
from src.visualization.plots import plot_roc_curve, plot_best_worst_videos
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Configuration
    test_dir = "/work3/s225224/ucf-crime/data/Test"
    annotation_path = "/work3/s225224/ucf-crime/data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    checkpoint_path = (
        "/work3/s225224/ucf-crime/checkpoints/r3d_binary_20251201_135422/best_model.pth"
    )
    plot_dir = "/work3/s225224/ucf-crime/experiments/plots"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=== Frame-Level Evaluation ===")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Load model
    model = create_r3d_classifier(
        num_classes=2, pretrained=False, freeze_backbone=True, dropout=0.5
    )
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    # Log checkpoint info
    if checkpoint:
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
        if "best_auc" in checkpoint:
            logger.info(f"Best training AUC: {checkpoint['best_auc']:.4f}")

    # Transform
    transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)

    # Evaluate
    metrics, video_results = evaluate_frame_level(
        model,
        test_dir,
        annotation_path,
        transform,
        device,
        clip_len=16,
        stride=16,
        sigma=5,
    )

    logger.info(f"\n{metrics}")

    # Plot results
    plot_dir = Path(plot_dir)
    plot_roc_curve(
        metrics.fpr, metrics.tpr, metrics.frame_auc, plot_dir / "roc_curve.png"
    )
    plot_best_worst_videos(video_results, {}, plot_dir, top_n=5)

    logger.info(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
