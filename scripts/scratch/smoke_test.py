import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import sys

# Import your dataset class
# Ensure your dataset file is named dataset.py, or change this import
from src.datasets.ucf import UCFCrimeDataset


def test_dry_run(root_dir):
    """
    Quickly validates directory structure and regex parsing
    WITHOUT loading images.
    """
    print(f"\n{'=' * 20} DRY RUN: FILE PARSING {'=' * 20}")

    # Initialize without transform to verify logic only
    try:
        ds = UCFCrimeDataset(root_dir, split="train", mode="binary")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not initialize dataset. {e}")
        return False

    print(f"✅ Found {len(ds)} clips total.")

    if len(ds) == 0:
        print("❌ Error: Dataset length is 0. Check root_dir path or folder structure.")
        return False

    # Check first sample metadata
    sample = ds.samples[0]
    print("Sample 0 Metadata:")
    print(f"  - Video ID: {sample.get('video_id', 'N/A')}")
    print(f"  - Label: {sample['label']}")
    print(f"  - Frames: {len(sample['paths'])} (Expected: {ds.clip_len})")
    print(f"  - Path Ex: {sample['paths'][0]}")

    # Check for correct regex grouping (Crucial!)
    # We want to ensure we didn't mix frames from different videos
    unique_vids = set(s["video_id"] for s in ds.samples)
    print(f"✅ Unique Videos Found: {len(unique_vids)}")

    return True


def test_live_load(root_dir):
    """Actually loads images and creates a batch."""
    print(f"\n{'=' * 20} LIVE RUN: TENSOR SHAPES {'=' * 20}")

    # Import transform
    from src.datasets.transforms import RGBVideoTransform

    transform = RGBVideoTransform(mode="train", crop_size=112, resize_size=128)
    ds = UCFCrimeDataset(
        root_dir,
        split="train",
        clip_len=16,
        mode="binary",
        transform=transform,  # Pass transform explicitly
    )

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    print("⏳ Loading first batch...")
    start_time = time.time()

    try:
        frames, labels = next(iter(loader))
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        import traceback

        traceback.print_exc()
        return

    load_time = time.time() - start_time
    print(f"✅ Batch loaded in {load_time:.2f} seconds.")

    # Check Shapes
    print("\n[Shape Check]")
    print(f"  Input Batch: {frames.shape}")  # Should be (4, 3, 16, 112, 112)
    print(f"  Labels: {labels.shape}")

    if frames.shape == torch.Size([4, 3, 16, 112, 112]):
        print("✅ Perfect! Upsampled to 112×112")
    else:
        print(f"❌ WARN: Expected (4, 3, 16, 112, 112), got {frames.shape}")

    # Check normalization (should be centered around 0 now, not 0-1)
    print("\n[Value Range Check - Kinetics Normalized]")
    print(f"  Mean: {frames.mean():.4f} (should be ~0)")
    print(f"  Std: {frames.std():.4f} (should be ~1)")
    print(f"  Max: {frames.max():.4f}, Min: {frames.min():.4f}")


def test_visual_sanity(root_dir):
    """
    Saves a grid of the first clip to visually confirm correct order.
    """
    print(f"\n{'=' * 20} VISUAL SANITY CHECK {'=' * 20}")
    ds = UCFCrimeDataset(root_dir, split="train", clip_len=16, stride=2)

    # Get a sample
    frames, label = ds[0]  # (C, T, H, W)

    # Convert back to (T, H, W, C) for plotting
    frames = frames.permute(1, 2, 3, 0).numpy()

    # Save a montage
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        if i < 16:
            # Denormalize if needed, here assuming 0-1
            img = (frames[i] * 255).astype("uint8")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Fr {i}")

    out_path = "sanity_check_clip.png"
    plt.savefig(out_path)
    print(f"✅ Saved visual check to {out_path}. Inspect this file!")
    print("   Verify that frames are consecutive and not shuffled.")


if __name__ == "__main__":
    # SET YOUR PATH HERE
    ROOT_DIR = "/work3/s225224/ucf-crime/data"

    if not os.path.exists(ROOT_DIR):
        print(f"❌ Path not found: {ROOT_DIR}")
        sys.exit(1)

    # 1. Logic Check
    success = test_dry_run(ROOT_DIR)

    # 2. Tensor Check
    if success:
        test_live_load(ROOT_DIR)

        # 3. Visual Check (Optional)
        # test_visual_sanity(ROOT_DIR)
