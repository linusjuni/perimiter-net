from src.utils.training_utils import AverageMeter, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train model for one epoch."""
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc = accuracy(outputs, targets)

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

    logger.info(
        f"Epoch [{epoch}] Training - Loss: {losses.avg:.4f} Acc: {accs.avg:.2f}%"
    )

    return losses.avg, accs.avg
