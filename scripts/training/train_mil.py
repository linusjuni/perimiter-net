import sys
import torch
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    """Main training function for MIL model."""

    mode = "motion"

    # Configuration
    feature_dir = f"/work3/s225224/ucf-crime/features/{mode}/Train"
    base_checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints/mil"

    val_split = 0.2

    input_dim = 512
    lr = 1e-4
    weight_decay = 5e-3
    epochs = 2000
    batch_size = 30
    segments = 32
    patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mil_{mode}_{timestamp}"
    checkpoint_dir = Path(base_checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"MIL Training Run: {run_name}")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Data Loading
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

    num_train_videos = len(train_loader.normal_videos) + len(
        train_loader.anomaly_videos
    )
    num_val_videos = len(val_loader.normal_videos) + len(val_loader.anomaly_videos)
    logger.info(f"Number of training videos: {num_train_videos}")
    logger.info(f"Number of validation videos: {num_val_videos}")

    # Model Setup
    logger.info("Initializing model...")
    model = MILModel(input_dim=input_dim).to(device)
    criterion = MILRankingLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=patience, mode="min")
    history = MILTrainingHistory(save_dir=checkpoint_dir)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Training Loop
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_loss = float("inf")
    start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_epoch_mil(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            batch_size=batch_size,
        )

        if epoch % 10 == 0 or epoch == 1:
            val_metrics = evaluate_mil(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                split="val",
            )

            history.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
            )

            is_best = val_metrics.loss < best_loss
            if is_best:
                best_loss = val_metrics.loss
                logger.info(f"🎯 New Best Loss: {best_loss:.4f}")

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_dir=checkpoint_dir,
                filename="checkpoint_epoch",
                is_best=is_best,
            )

            early_stopping(val_metrics.loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    # Training Complete
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best Validation Loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    best_epoch_info = history.get_best_epoch(metric="val_loss")
    if best_epoch_info:
        logger.info(f"Best epoch: {best_epoch_info['epoch']}")
        logger.info(f"Best metrics: {best_epoch_info}")


if __name__ == "__main__":
    main()
