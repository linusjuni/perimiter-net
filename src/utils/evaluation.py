import torch
from src.utils.training_utils import AverageMeter, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(model, val_loader, criterion, device, split="val"):
    """Evaluate model on validation or test set."""
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute accuracy
            acc = accuracy(outputs, targets)

            # Update metrics
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            accs.update(acc.item(), batch_size)

    logger.info(f"{split.capitalize()} - Loss: {losses.avg:.4f} Acc: {accs.avg:.2f}%")

    return losses.avg, accs.avg
