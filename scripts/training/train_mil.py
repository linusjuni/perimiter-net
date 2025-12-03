import sys
import torch
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel
from src.datasets.mil import MILDataLoader
from src.utils.losses import MILRankingLoss
from src.utils.training import train_epoch_mil
from src.utils.mil_evaluation import evaluate_mil
from src.utils.training_utils import (
    save_checkpoint,
    EarlyStopping,
    MILTrainingHistory,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # --- Configuration ---
    feature_dir = "/work3/s225224/ucf-crime/features/rgb/Train"  # Single directory
    base_checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints/mil"

    # Train/Val split ratio
    val_split = 0.2  # 20% for validation

    # Hyperparameters
    input_dim = 512
    lr = 1e-3
    weight_decay = 0.005
    epochs = 2000
    batch_size = 30  # Videos per class (total batch = 60)
    segments = 32
    patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mil_rgb_{timestamp}"
    checkpoint_dir = Path(base_checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"MIL Training Run: {run_name}")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # --- Data Loading ---
    logger.info("Loading datasets...")
    train_loader = MILDataLoader(
        feature_dir=feature_dir,
        segments=segments,
        shuffle=True,
        split="train",
        val_split=val_split,
    )
    val_loader = MILDataLoader(
        feature_dir=feature_dir,
        segments=segments,
        shuffle=False,
        split="val",
        val_split=val_split,
    )

    # Number of videos
    num_train_videos = len(train_loader.normal_videos) + len(
        train_loader.anomaly_videos
    )
    num_val_videos = len(val_loader.normal_videos) + len(val_loader.anomaly_videos)
    logger.info(f"Number of training videos: {num_train_videos}")
    logger.info(f"Number of validation videos: {num_val_videos}")

    # --- Model Setup ---
    logger.info("Initializing model...")
    model = MILModel(input_dim=input_dim).to(device)
    criterion = MILRankingLoss()

    # Optimizer and Scheduler (consistent with R3D)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training utilities
    early_stopping = EarlyStopping(patience=patience, mode="max")
    history = MILTrainingHistory(save_dir=checkpoint_dir)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --- Training Loop ---
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_auc = 0.0
    start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_metrics = train_epoch_mil(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            batch_size=batch_size,
        )

        # Validate (every 50 epochs to save time)
        if epoch % 50 == 0 or epoch == 1:
            val_metrics = evaluate_mil(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                split="val",
            )

            # Update history
            history.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
            )

            # Save checkpoint
            is_best = val_metrics.auc > best_auc
            if is_best:
                best_auc = val_metrics.auc
                logger.info(f"🎯 New Best AUC: {best_auc:.4f}")

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_auc": best_auc,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch}.pth",
                is_best=is_best,
            )

            # Early stopping
            early_stopping(val_metrics.auc)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Step scheduler
        scheduler.step()

    # --- Training Complete ---
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best Validation AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    # Get best epoch info
    best_epoch_info = history.get_best_epoch(metric="val_auc")
    if best_epoch_info:
        logger.info(f"Best epoch: {best_epoch_info['epoch']}")
        logger.info(f"Best metrics: {best_epoch_info}")


if __name__ == "__main__":
    main()
