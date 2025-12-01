import time
import torch
from torch.amp import autocast
from src.utils.training_utils import AverageMeter, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train model for one epoch."""
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # --- 2. Forward pass with AMP ---
        # 'enabled' checks if scaler was passed. If None, it runs in standard float32.
        with autocast("cuda", enabled=(scaler is not None)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # --- 3. Backward pass with Scaler ---
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Fallback for standard FP32 training
            loss.backward()
            optimizer.step()

        # Compute accuracy (detach outputs to save memory during calculation)
        with torch.no_grad():
            acc = accuracy(outputs.detach(), targets)

        # Update metrics
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(acc.item(), batch_size)

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
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
