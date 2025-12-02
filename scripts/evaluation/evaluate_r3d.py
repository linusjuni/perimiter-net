import sys
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.r3d import create_r3d_classifier
from src.datasets.transforms import RGBVideoTransform
from src.utils.frame_level_evaluation import (
    evaluate_frame_level,
    save_frame_level_results,
)
from src.utils.training_utils import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_available_checkpoints(base_dir="/work3/s225224/ucf-crime/checkpoints"):
    """Find all available model checkpoints."""
    base_path = Path(base_dir)
    checkpoints = []

    for run_dir in sorted(base_path.glob("r3d_*")):
        if run_dir.is_dir():
            best_model = run_dir / "best_model.pth"
            if best_model.exists():
                checkpoints.append(best_model)

    return checkpoints


def select_checkpoint(checkpoints):
    """Interactive checkpoint selection."""
    if not checkpoints:
        logger.error("No checkpoints found!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Available Checkpoints:")
    print("=" * 60)
    for i, ckpt in enumerate(checkpoints, 1):
        run_name = ckpt.parent.name
        print(f"  [{i}] {run_name}")
    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid selection. Exiting.")
            sys.exit(0)


def main():
    # Fixed paths
    test_dir = "/work3/s225224/ucf-crime/data/Test"
    annotation_path = "/work3/s225224/ucf-crime/data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    results_base_dir = Path("/work3/s225224/ucf-crime/experiments/frame_level")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Interactive checkpoint selection
    checkpoints = find_available_checkpoints()
    checkpoint_path = select_checkpoint(checkpoints)

    # Create experiment-specific results directory
    run_name = checkpoint_path.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #results_dir = results_base_dir / f"{run_name}_{timestamp}" # choose this
    plot_dir = Path("results") # or this
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Frame-Level Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Results dir: {results_dir}")
    logger.info("=" * 60)

    # Load model
    model = create_r3d_classifier(
        num_classes=2, pretrained=False, freeze_backbone=True, dropout=0.5
    )
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.to(device).eval()

    if checkpoint:
        epoch = checkpoint.get("epoch", "unknown")
        best_auc = checkpoint.get("best_auc", 0.0)
        logger.info(f"Loaded epoch {epoch} | Training AUC: {best_auc:.4f}")

    # Evaluation settings
    transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)

    # Run evaluation
    metrics, curves, video_results, all_scores, all_labels = evaluate_frame_level(
        model,
        test_dir,
        annotation_path,
        transform,
        device,
        clip_len=16,
        stride=8,
        sigma=5,
    )

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"{metrics}")
    logger.info("=" * 60)

    save_frame_level_results(
        results_dir=results_dir,
        run_name=run_name,
        checkpoint_path=checkpoint_path,
        timestamp=timestamp,
        metrics=metrics,
        curves=curves,
        scores=all_scores,
        labels=all_labels,
        video_results=video_results,
    )

    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Run plotting script to generate visualizations:")
    logger.info(f"  python scripts/plotting/plot_r3d_results.py {results_dir}")


if __name__ == "__main__":
    main()
