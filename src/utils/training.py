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


def train_epoch_mil(
    model: torch.nn.Module,
    loader,  # MILDataLoader
    criterion,  # MILRankingLoss
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
) -> dict:
    """
    Train MIL model for one epoch.
    """
    model.train()

    # Meters for each loss component
    loss_meter = AverageMeter()
    rank_meter = AverageMeter()
    sparsity_meter = AverageMeter()
    smoothness_meter = AverageMeter()

    start_time = time.time()

    # Calculate number of batches
    num_batches = loader.get_num_batches(batch_size)

    for batch_idx in range(num_batches):
        # Get balanced batch
        norm_batch, anom_batch = loader.get_batch(batch_size)
        norm_batch = norm_batch.to(device, non_blocking=True)
        anom_batch = anom_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        norm_preds = model(norm_batch)  # (batch_size, 32, 1)
        anom_preds = model(anom_batch)  # (batch_size, 32, 1)

        # Compute loss
        total_loss, loss_components = criterion(norm_preds, anom_preds)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(total_loss.item(), batch_size)
        rank_meter.update(loss_components["rank"], batch_size)
        sparsity_meter.update(loss_components["sparse"], batch_size)
        smoothness_meter.update(loss_components["smooth"], batch_size)

    return {
        "loss": loss_meter.avg,
        "rank_loss": rank_meter.avg,
        "sparsity_loss": sparsity_meter.avg,
        "smoothness_loss": smoothness_meter.avg,
    }
