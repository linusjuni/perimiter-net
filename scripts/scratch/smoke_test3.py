"""
Smoke test for Sobel Motion Transform visualization.
Loads a few video clips and plots the dx, dy, dt channels.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.ucf import UCFCrimeDataset
from src.datasets.transforms import SobelMotionTransform


def visualize_motion_clip(clip_tensor, frame_paths, sample_idx=0, class_name="Unknown"):
    """
    Visualize motion overlay only - no text, no axes, just the image.

    Args:
        clip_tensor: (3, T, H, W) tensor with [dx, dy, dt]
        frame_paths: List of paths to original frames
        sample_idx: which sample this is (for plot title)
        class_name: Name of the class (e.g., "Normal", "Assault", etc.)
    """
    # Convert to numpy and select middle frame
    clip_np = clip_tensor.cpu().numpy()
    C, T, H, W = clip_np.shape
    mid_frame = T // 2

    dx = clip_np[0, mid_frame]
    dy = clip_np[1, mid_frame]
    dt = clip_np[2, mid_frame]

    # Load original frame at FULL RESOLUTION (no resize)
    orig_frame = cv2.imread(frame_paths[mid_frame])
    orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_frame.shape[:2]

    # Combined magnitude
    magnitude = np.sqrt(dx**2 + dy**2 + dt**2)
    # Normalize magnitude to [0, 1] for visualization
    magnitude_norm = (magnitude - magnitude.min()) / (
        magnitude.max() - magnitude.min() + 1e-8
    )

    # Resize motion magnitude to match original frame resolution
    magnitude_fullres = cv2.resize(
        magnitude_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    # Brighten the original frame slightly
    brightened_frame = np.clip(orig_frame * 1.2, 0, 255).astype(np.uint8)

    # Create figure with exact size of image (DPI = 100 for 1:1 pixel mapping)
    dpi = 100
    fig = plt.figure(figsize=(orig_w / dpi, orig_h / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot overlay
    ax.imshow(brightened_frame)
    ax.imshow(magnitude_fullres, cmap="jet", alpha=0.6)

    return fig


def print_statistics(clip_tensor, label, class_name):
    """Print basic statistics about the motion features."""
    clip_np = clip_tensor.cpu().numpy()

    print(f"\n{'=' * 60}")
    print(f"Class: {class_name} | Label: {label}")
    print(f"Shape: {clip_np.shape} (C, T, H, W)")
    print(f"{'=' * 60}")

    for i, channel_name in enumerate(["dx", "dy", "dt"]):
        channel_data = clip_np[i]
        print(f"\n{channel_name}:")
        print(f"  Min:  {channel_data.min():.4f}")
        print(f"  Max:  {channel_data.max():.4f}")
        print(f"  Mean: {channel_data.mean():.4f}")
        print(f"  Std:  {channel_data.std():.4f}")

        # Count non-zero (motion detection)
        non_zero = np.count_nonzero(np.abs(channel_data) > 0.01)
        total = channel_data.size
        print(f"  Motion Pixels: {non_zero}/{total} ({100 * non_zero / total:.1f}%)")


def main():
    # Configuration
    root_dir = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Frames"  # Update if testing locally
    clip_len = 16
    num_samples = 5  # Number of clips to visualize

    print("=" * 60)
    print("SOBEL MOTION TRANSFORM SMOKE TEST")
    print("=" * 60)

    # Initialize dataset with motion transform
    print("\nLoading dataset...")
    transform = SobelMotionTransform(mode="val", crop_size=112, resize_size=128)

    dataset = UCFCrimeDataset(
        root_dir=root_dir,
        split="train",
        clip_len=clip_len,
        transform=transform,
        stride=clip_len,
        val_ratio=0.20,
    )

    print(f"Dataset loaded: {len(dataset)} clips")

    # Load and visualize samples
    print(f"\nLoading {num_samples} samples...")

    # Try to get both normal and anomaly samples
    anomaly_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 1]
    normal_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]

    # Select diverse samples - alternate between anomaly and normal
    test_indices = []
    anomaly_step = max(1, len(anomaly_indices) // (num_samples // 2 + 1))
    normal_step = max(1, len(normal_indices) // (num_samples // 2 + 1))

    for i in range(num_samples):
        if i % 2 == 0 and len(anomaly_indices) > 0:
            # Add anomaly sample
            idx = (i // 2) * anomaly_step
            if idx < len(anomaly_indices):
                test_indices.append(anomaly_indices[idx])
        else:
            # Add normal sample
            idx = (i // 2) * normal_step
            if idx < len(normal_indices):
                test_indices.append(normal_indices[idx])

    for sample_num, idx in enumerate(test_indices):
        print(f"\n--- Sample {sample_num + 1}/{len(test_indices)} ---")

        # Get frame paths before transformation
        frame_paths = dataset.samples[idx]["paths"]

        # Load clip
        clip, label = dataset[idx]

        # Get class name from the frame path
        class_folder = Path(frame_paths[0]).parent.name

        # Print statistics
        print_statistics(clip, label, class_folder)

        # Visualize
        fig = visualize_motion_clip(
            clip, frame_paths, sample_idx=sample_num + 1, class_name=class_folder
        )

        # Save figure
        output_path = (
            Path(__file__).parent.parent.parent
            / f"motion_test_sample_{sample_num + 1}.png"
        )
        fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"\nSaved visualization to: {output_path}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print("\nCheck the generated PNG files to verify:")
    print("  - Motion overlay highlights moving regions")
    print("  - Anomaly videos should have more intense motion")


if __name__ == "__main__":
    main()
