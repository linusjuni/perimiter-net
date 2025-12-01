import torch
import matplotlib.pyplot as plt
import torchvision
from src.datasets.ucf import UCFCrimeDataset  # Ensure this matches your filename


def save_sanity_check(output_file="sanity_check.png"):
    # 1. Init Dataset (Force 224x224 for visibility)
    # We define a simple transform here to ensure we aren't seeing 64x64
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    ds = UCFCrimeDataset(
        root_dir="/work3/s225224/ucf-crime/data",
        split="train",
        clip_len=16,
        stride=16,  # Use high stride to load faster for this test
        transform=None,  # We will handle transform manually below for control
    )

    # 2. Get a sample (Try to find an Anomaly, not Normal)
    print("Searching for an anomaly clip to visualize...")
    for i in range(len(ds)):
        sample = ds.samples[i]
        if sample["label"] == 0:  # Found a Normal clip
            print(f"Found Normal: {sample['video_id']}")
            frames, label = ds[i]
            break

    # frames is (C, T, H, W) -> e.g. (3, 16, 224, 224)
    # 3. Create a Grid
    # We need to rearrange to (T, C, H, W) for make_grid
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(1, 0, 2, 3)

    # Create grid of 16 frames (4 rows of 4)
    grid_img = torchvision.utils.make_grid(frames, nrow=4, padding=2)

    # 4. Save
    plt.figure(figsize=(12, 12))
    # Permute (C, H, W) -> (H, W, C) for Matplotlib
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Label: {label} (Anomaly) - Video: {sample['video_id']}")
    plt.savefig(output_file)
    print(f"✅ Saved sanity check to {output_file}")


if __name__ == "__main__":
    save_sanity_check()
