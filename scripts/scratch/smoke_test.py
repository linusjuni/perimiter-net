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
    print(f"Sample 0 Metadata:")
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
    """
    Actually loads images and creates a batch to verify
    transforms and tensor shapes.
    """
    print(f"\n{'=' * 20} LIVE RUN: TENSOR SHAPES {'=' * 20}")

    ds = UCFCrimeDataset(root_dir, split="train", clip_len=16, mode="binary")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    print("⏳ Loading first batch... (this may take a moment)")
    start_time = time.time()

    try:
        frames, labels = next(iter(loader))
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        return

    load_time = time.time() - start_time
    print(f"✅ Batch loaded in {load_time:.2f} seconds.")

    # 1. Check Shapes
    # Expected: (Batch, Channels, Time, Height, Width) -> (4, 3, 16, H, W)
    print(f"\n[Shape Check]")
    print(f"  Input Batch: {frames.shape}")
    print(f"  Labels: {labels.shape}")

    if frames.dim() == 5 and frames.shape[2] == 16:
        print("✅ Temporal dimension is correct (16 frames).")
    else:
        print("❌ WARN: Unexpected shape. Ensure (B, C, T, H, W).")

    # 2. Check Value Range
    print(f"\n[Value Range Check]")
    print(f"  Max: {frames.max():.4f}, Min: {frames.min():.4f}")
    if frames.max() <= 1.0 and frames.min() >= 0.0:
        print("✅ Normalization appears correct (0-1).")
    else:
        print("❌ WARN: Values outside [0, 1]. Check your transforms.")

    # 3. Check Binary Labels
    print(f"\n[Label Check]")
    print(f"  Labels found: {labels.tolist()}")
    if ds.mode == "binary":
        if all(l in [0, 1] for l in labels):
            print("✅ Binary labels are correct (0s and 1s).")
        else:
            print("❌ Error: Found non-binary labels in binary mode.")


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
