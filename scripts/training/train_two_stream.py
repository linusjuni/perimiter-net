import sys
import torch
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import TwoStreamMIL
from src.datasets.mil import TwoStreamMILDataLoader
from src.utils.losses import MILRankingLoss
from src.utils.training import train_epoch_two_stream
from src.utils.mil_evaluation import evaluate_two_stream
from src.utils.training_utils import (
    save_checkpoint,
    EarlyStopping,
    MILTrainingHistory,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # --- Configuration ---
    rgb_feature_dir = "/work3/s225224/ucf-crime/features/rgb/Train"
    motion_feature_dir = "/work3/s225224/ucf-crime/features/motion/Train"
    base_checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints/two_stream_mil"

    # Fusion mode: 'early' or 'late'
    fusion_mode = "early"

    # For late fusion only (set to None for early fusion)
    rgb_checkpoint = (
        None  # e.g., "/work3/s225224/ucf-crime/checkpoints/mil/rgb_best/best_model.pth"
    )
    motion_checkpoint = None  # e.g., "/work3/s225224/ucf-crime/checkpoints/mil/motion_best/best_model.pth"
    freeze_streams = True

    # Train/Val split ratio
    val_split = 0.2

    # Hyperparameters
    input_dim = 512
    lr = 1e-3
    weight_decay = 0.005
    epochs = 2000
    batch_size = 30
    segments = 32
    patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"two_stream_{fusion_mode}_{timestamp}"
    checkpoint_dir = Path(base_checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Two-Stream MIL Training Run: {run_name}")
    logger.info("=" * 80)
    logger.info(f"Fusion Mode: {fusion_mode}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Data Loading ---
    logger.info("Loading datasets...")
    train_loader = TwoStreamMILDataLoader(
        rgb_feature_dir=rgb_feature_dir,
        motion_feature_dir=motion_feature_dir,
        segments=segments,
        shuffle=True,
        split="train",
        val_split=val_split,
    )
    val_loader = TwoStreamMILDataLoader(
        rgb_feature_dir=rgb_feature_dir,
        motion_feature_dir=motion_feature_dir,
        segments=segments,
        shuffle=False,
        split="val",
        val_split=val_split,
    )

    num_train_videos = len(train_loader.normal_videos) + len(
        train_loader.anomaly_videos
    )
    num_val_videos = len(val_loader.normal_videos) + len(val_loader.anomaly_videos)
    logger.info(f"Training videos: {num_train_videos}")
    logger.info(f"Validation videos: {num_val_videos}")

    # --- Model Setup ---
    logger.info("Initializing model...")
    model = TwoStreamMIL(
        input_dim=input_dim,
        fusion_mode=fusion_mode,
        rgb_checkpoint=rgb_checkpoint,
        motion_checkpoint=motion_checkpoint,
        freeze_streams=freeze_streams,
    ).to(device)

    criterion = MILRankingLoss()

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training utilities
    early_stopping = EarlyStopping(patience=patience, mode="max")
    history = MILTrainingHistory(save_dir=checkpoint_dir)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    # --- Training Loop ---
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_metrics = train_epoch_two_stream(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            batch_size=batch_size,
        )

        # Validate
        val_metrics = evaluate_two_stream(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split="val",
        )

        scheduler.step()

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
            logger.info(f"New best AUC: {best_auc:.4f}")

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "fusion_mode": fusion_mode,
            },
            checkpoint_dir=checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch}.pth",
            is_best=is_best,
        )

        # Early stopping
        if early_stopping(val_metrics.auc):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # --- Training Complete ---
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best Validation AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    best_epoch_info = history.get_best_epoch(metric="val_auc")
    if best_epoch_info:
        logger.info(f"Best epoch: {best_epoch_info['epoch']}")


if __name__ == "__main__":
    main()
