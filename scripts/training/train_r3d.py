import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets.ucf import UCFCrimeDataset  # Changed from VIRATDataset
from src.datasets.transforms import RGBVideoTransform
from src.models.r3d import create_r3d_classifier
from src.utils.training_utils import save_checkpoint, load_checkpoint, EarlyStopping
from src.utils.training import train_epoch
from src.utils.evaluation import evaluate
from src.utils.logger import get_logger
from src.utils.losses import FocalLoss  # Added import


def main():
    # --- Config ---
    root_dir = "/work3/s225224/ucf-crime/data"  # Changed path
    checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints"  # Changed path
    num_classes = 2  # Changed from 7 to 2 (binary: Normal vs Anomaly)
    batch_size = 16
    num_workers = 4
    num_epochs = 10  # Changed from 30 to 10 for baseline
    lr = 1e-4
    weight_decay = 1e-2
    patience = 5  # Changed from 7 to 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_len = 16

    logger = get_logger(__name__)

    # --- Data ---
    logger.info("Setting up datasets and dataloaders...")
    train_transform = RGBVideoTransform(mode="train", crop_size=112, resize_size=128)
    val_transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)

    train_dataset = UCFCrimeDataset(  # Changed from VIRATDataset
        root_dir, split="train", clip_len=clip_len, transform=train_transform
    )
    val_dataset = UCFCrimeDataset(  # Changed from VIRATDataset
        root_dir, split="val", clip_len=clip_len, transform=val_transform
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

    # Use Focal Loss with class weights: Normal=0.25, Anomaly=0.75
    alpha = torch.tensor([0.25, 0.75])  # Weight anomaly class higher
    criterion = FocalLoss(alpha=alpha, gamma=2.0)  # Changed from CrossEntropyLoss

    # --- Early Stopping ---
    early_stopper = EarlyStopping(patience=patience, mode="max")

    # --- Optionally resume from checkpoint ---
    start_epoch = 0
    best_acc = 0.0
    resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(resume_path):
        logger.info("Resuming from checkpoint...")
        start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, device)

    # --- Training Loop ---
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, split="val")

        # Scheduler step
        scheduler.step()

        # Save checkpoint
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        save_checkpoint(state, checkpoint_dir, is_best=val_acc > best_acc)
        if val_acc > best_acc:
            best_acc = val_acc

        # Early stopping
        if early_stopper(val_acc):
            logger.warning("Early stopping triggered. Training halted.")
            break

    logger.info(f"Training complete. Best val accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
