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


def train_epoch_two_stream(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
) -> dict:
    """Train two-stream MIL model for one epoch."""
    model.train()

    losses = AverageMeter()
    rank_losses = AverageMeter()
    sparse_losses = AverageMeter()
    smooth_losses = AverageMeter()

    num_batches = loader.get_num_batches(batch_size)

    for batch_idx in range(num_batches):
        # Get paired batch: (normal_rgb, normal_motion, anomaly_rgb, anomaly_motion)
        norm_rgb, norm_motion, anom_rgb, anom_motion = loader.get_batch(batch_size)

        norm_rgb = norm_rgb.to(device)
        norm_motion = norm_motion.to(device)
        anom_rgb = anom_rgb.to(device)
        anom_motion = anom_motion.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds_normal = model(norm_rgb, norm_motion)
        preds_anomaly = model(anom_rgb, anom_motion)

        # Compute loss
        loss, loss_components = criterion(preds_normal, preds_anomaly)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), batch_size)
        rank_losses.update(loss_components["rank"], batch_size)
        sparse_losses.update(loss_components["sparse"], batch_size)
        smooth_losses.update(loss_components["smooth"], batch_size)

    metrics = {
        "loss": losses.avg,
        "rank_loss": rank_losses.avg,
        "sparse_loss": sparse_losses.avg,
        "smooth_loss": smooth_losses.avg,
    }

    logger.info(
        f"Epoch {epoch} | Train Loss: {losses.avg:.4f} "
        f"(Rank: {rank_losses.avg:.4f}, Sparse: {sparse_losses.avg:.6f}, Smooth: {smooth_losses.avg:.6f})"
    )

    return metrics