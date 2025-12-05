import torch
from src.datasets.virat import VIRATDataset
from src.datasets.transforms import RGBVideoTransform
from src.utils.logger import get_logger


def main():
    root_dir = "/work3/s225224/perimeter-net/data"
    num_classes = 7
    clip_len = 16

    logger = get_logger(__name__)
    logger.info("Computing class weights for imbalanced data...")

    train_transform = RGBVideoTransform(mode="train", crop_size=112, resize_size=128)
    train_dataset = VIRATDataset(
        root_dir, split="train", clip_len=clip_len, transform=train_transform
    )

    class_counts = torch.zeros(num_classes)
    for _, label in train_dataset:
        class_counts[label] += 1

    logger.info(f"Class counts: {class_counts.tolist()}")

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    logger.info(f"Class weights: {class_weights.tolist()}")


if __name__ == "__main__":
    main()
