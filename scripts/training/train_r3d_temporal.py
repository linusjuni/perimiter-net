from datetime import datetime
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.ucf import UCFCrimeDataset

# IMPORT THE MOTION TRANSFORM
from src.datasets.transforms import SobelMotionTransform
from src.models.r3d import create_r3d_classifier
from src.utils.training_utils import (
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    TrainingHistory,
)
from src.utils.training import train_epoch
from src.utils.clip_level_evaluation import evaluate
from src.utils.logger import get_logger
from src.utils.losses import FocalLoss


def main():
    # --- Config ---
    # IMPORTANT: Ensure this points to the folder containing your frames
    # (e.g. /dtu/blackhole/.../Frames or /work3/.../data if you moved them)
    root_dir = "/work3/s225224/ucf-crime/data"
    base_checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints"

    # Create model-specific directory with timestamp
    # Changed name to indicate Motion Stream
    model_name = "r3d_motion_sobel"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    checkpoint_dir = Path(base_checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 2
    batch_size = 128
    num_workers = 4
    num_epochs = 20
    # Hyperparameters from your RGB script
    lr = 1e-5
    weight_decay = 1e-1
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_len = 16

    logger = get_logger(__name__)
    logger.info(f"Starting MOTION STREAM training run: {run_name}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- Training History Tracker ---
    history = TrainingHistory(checkpoint_dir)

    # --- Data ---
    logger.info("Setting up datasets with SOBEL MOTION Transform...")

    # USE SOBEL MOTION TRANSFORM
    # This calculates dx, dy, dt derivatives on the fly
    train_transform = SobelMotionTransform(mode="train", crop_size=112, resize_size=128)
    val_transform = SobelMotionTransform(mode="val", crop_size=112, resize_size=128)

    # Use 20% of the Training folder for Validation
    val_ratio = 0.20

    train_dataset = UCFCrimeDataset(
        root_dir,
        split="train",
        clip_len=clip_len,
        transform=train_transform,  # Pass motion transform
        stride=clip_len,
        val_ratio=val_ratio,
    )

    val_dataset = UCFCrimeDataset(
        root_dir,
        split="val",
        clip_len=clip_len,
        transform=val_transform,  # Pass motion transform
        stride=clip_len,
        val_ratio=val_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Model ---
    logger.info("Instantiating R3D-18 (Motion Stream)...")
    # Pretrained weights are still useful for motion (shapes are similar)
    model = create_r3d_classifier(
        num_classes=num_classes, pretrained=True, freeze_backbone=False, dropout=0.7
    )
    model = model.to(device)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Use Focal Loss with class weights
    alpha = torch.tensor([0.25, 0.75])
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    criterion = criterion.to(device)

    # --- AMP Scaler ---
    scaler = GradScaler("cuda")

    # --- Early Stopping ---
    early_stopper = EarlyStopping(patience=patience, mode="max")

    # --- Resume ---
    start_epoch = 0
    best_auc = 0.0
    resume_path = checkpoint_dir / "checkpoint.pth"
    if resume_path.exists():
        logger.info("Resuming from checkpoint...")
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0)
        best_auc = checkpoint.get("best_auc", 0.0)

    # --- Log Device and Environment ---
    logger.info(f"Training on device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # --- Training Loop ---
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch + 1,
            scaler=scaler,
        )

        # Evaluate (Motion Stream)
        val_metrics = evaluate(model, val_loader, criterion, device, split="val")

        # Extract metrics
        val_loss = val_metrics.loss
        val_acc = val_metrics.acc
        val_auc = val_metrics.auc

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update training history
        history.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_metrics=val_metrics,
            learning_rate=current_lr,
        )

        # Log validation results
        logger.info(
            f"Epoch [{epoch + 1}] Motion Validation - "
            f"Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}"
        )
        logger.info(
            f"  Motion Anomaly Stats: P={val_metrics.anomaly_precision:.3f} "
            f"R={val_metrics.anomaly_recall:.3f} F1={val_metrics.anomaly_f1:.3f}"
        )

        if val_metrics.tp > 0:
            logger.info(
                f"  Conf Matrix: TP={val_metrics.tp} FN={val_metrics.fn} "
                f"TN={val_metrics.tn} FP={val_metrics.fp}"
            )

        scheduler.step()

        is_best = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            logger.info(f"🔥 New Best Motion AUC: {val_auc:.4f}")

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "val_metrics": val_metrics.to_dict(),
        }
        save_checkpoint(state, checkpoint_dir, is_best=is_best)

        if early_stopper(val_auc):
            logger.warning("Early stopping triggered. Training halted.")
            break

    # Final summary
    best_epoch_info = history.get_best_epoch("val_auc")
    logger.info(
        f"Motion Training complete. Best AUC: {best_auc:.4f} at epoch {best_epoch_info['epoch']}"
    )
    logger.info(f"Training history saved to: {history.csv_path}")


if __name__ == "__main__":
    main()
