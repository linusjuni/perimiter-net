import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler  # <--- 1. IMPORT THIS

from src.datasets.ucf import UCFCrimeDataset
from src.datasets.transforms import RGBVideoTransform
from src.models.r3d import create_r3d_classifier
from src.utils.training_utils import save_checkpoint, load_checkpoint, EarlyStopping
from src.utils.training import train_epoch
from src.utils.evaluation import evaluate
from src.utils.logger import get_logger
from src.utils.losses import FocalLoss


def main():
    # --- Config ---
    root_dir = "/work3/s225224/ucf-crime/data"
    checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints"
    num_classes = 2
    batch_size = 32
    num_workers = 4
    num_epochs = 10
    lr = 1e-4
    weight_decay = 1e-2
    patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_len = 16

    logger = get_logger(__name__)

    # --- Data ---
    logger.info("Setting up datasets and dataloaders...")
    train_transform = RGBVideoTransform(mode="train", crop_size=112, resize_size=128)
    val_transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)

    train_dataset = UCFCrimeDataset(
        root_dir,
        split="train",
        clip_len=clip_len,
        transform=train_transform,
        stride=clip_len,
    )
    val_dataset = UCFCrimeDataset(
        root_dir,
        split="val",
        clip_len=clip_len,
        transform=val_transform,
        stride=clip_len,
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
    logger.info("Instantiating model...")
    model = create_r3d_classifier(
        num_classes=num_classes, pretrained=True, freeze_backbone=True, dropout=0.5
    )
    model = model.to(device)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

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
    best_auc = 0.0  # Changed from best_acc
    resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(resume_path):
        logger.info("Resuming from checkpoint...")
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0)
        best_auc = checkpoint.get("best_auc", 0.0)

    # --- Log Device ---
    logger.info(f"Training on device: {device}")

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

        # Evaluate and get metrics dataclass
        val_metrics = evaluate(model, val_loader, criterion, device, split="val")

        # Extract metrics
        val_loss = val_metrics.loss
        val_acc = val_metrics.acc
        val_auc = val_metrics.auc

        # Log validation results
        logger.info(
            f"Epoch [{epoch + 1}] Validation Summary - "
            f"Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}"
        )
        logger.info(
            f"  Anomaly Detection: Precision={val_metrics.anomaly_precision:.3f} "
            f"Recall={val_metrics.anomaly_recall:.3f} F1={val_metrics.anomaly_f1:.3f}"
        )

        # Log confusion matrix if binary
        if val_metrics.tp > 0:
            logger.info(
                f"  Confusion Matrix: TP={val_metrics.tp} FN={val_metrics.fn} "
                f"TN={val_metrics.tn} FP={val_metrics.fp}"
            )
            logger.info(
                f"  Error Rates: FPR={val_metrics.fpr:.4f} FNR={val_metrics.fnr:.4f}"
            )

        scheduler.step()

        is_best = val_auc > best_auc

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "val_metrics": val_metrics.to_dict(),
        }
        save_checkpoint(state, checkpoint_dir, is_best=is_best)

        if is_best:
            best_auc = val_auc
            logger.info(f"New best AUC: {val_auc:.4f}")

        if early_stopper(val_auc):
            logger.warning("Early stopping triggered. Training halted.")
            break

    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
