import time
import torch
from torch.amp import autocast
from src.utils.training_utils import AverageMeter, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler=None,
) -> tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Neural network to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number (for logging)
        scaler: Optional GradScaler for mixed precision training

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    start_time = time.time()

    # Adaptive logging interval based on dataset size
    log_interval = max(10, len(train_loader) // 100)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)  # non_blocking for efficiency
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast("cuda", enabled=(scaler is not None)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            acc = accuracy(outputs.detach(), targets)

        # Update metrics
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(acc.item(), batch_size)

        # Adaptive logging
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {losses.avg:.4f} Acc: {accs.avg:.2f}%"
            )

    duration = time.time() - start_time
    logger.info(
        f"Epoch [{epoch}] Training Completed in {duration:.0f}s - "
        f"Loss: {losses.avg:.4f} Acc: {accs.avg:.2f}%"
    )

    return losses.avg, accs.avg
