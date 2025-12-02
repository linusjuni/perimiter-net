"""
Smoke test for Sobel Motion Transform visualization.
Loads a few video clips and plots the dx, dy, dt channels.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.ucf import UCFCrimeDataset
from src.datasets.transforms import SobelMotionTransform


def visualize_motion_clip(clip_tensor, sample_idx=0):
    """
    Visualize a single motion clip with its three channels.

    Args:
        clip_tensor: (3, T, H, W) tensor with [dx, dy, dt]
        sample_idx: which sample this is (for plot title)
    """
    # Convert to numpy and select middle frame
    clip_np = clip_tensor.cpu().numpy()
    C, T, H, W = clip_np.shape
    mid_frame = T // 2

    dx = clip_np[0, mid_frame]  # Horizontal gradients
    dy = clip_np[1, mid_frame]  # Vertical gradients
    dt = clip_np[2, mid_frame]  # Temporal gradients

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Motion Features - Sample {sample_idx} - Frame {mid_frame}/{T}")

    # Plot each channel
    im0 = axes[0].imshow(dx, cmap="seismic", vmin=-1, vmax=1)
    axes[0].set_title("dx (Horizontal Motion)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(dy, cmap="seismic", vmin=-1, vmax=1)
    axes[1].set_title("dy (Vertical Motion)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(dt, cmap="seismic", vmin=-1, vmax=1)
    axes[2].set_title("dt (Temporal Change)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Combined magnitude
    magnitude = np.sqrt(dx**2 + dy**2 + dt**2)
    im3 = axes[3].imshow(magnitude, cmap="hot")
    axes[3].set_title("Motion Magnitude")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
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
    root_dir = "/work3/s225224/ucf-crime/data"  # Update if testing locally
    clip_len = 16
    num_samples = 3  # Number of clips to visualize

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

    # Select diverse samples
    test_indices = []
    if len(anomaly_indices) > 0:
        test_indices.append(anomaly_indices[0])
    if len(normal_indices) > 0:
        test_indices.append(normal_indices[0])
    if len(anomaly_indices) > 1:
        test_indices.append(anomaly_indices[len(anomaly_indices) // 2])

    test_indices = test_indices[:num_samples]

    for sample_num, idx in enumerate(test_indices):
        print(f"\n--- Sample {sample_num + 1}/{num_samples} ---")

        # Load clip
        clip, label = dataset[idx]

        # Get class name
        class_name = "Normal" if label == 0 else "Anomaly"

        # Print statistics
        print_statistics(clip, label, class_name)

        # Visualize
        fig = visualize_motion_clip(clip, sample_idx=sample_num + 1)

        # Save figure
        output_path = Path(__file__).parent / f"motion_test_sample_{sample_num + 1}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved visualization to: {output_path}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print("\nCheck the generated PNG files to verify:")
    print("  - dx shows horizontal motion edges")
    print("  - dy shows vertical motion edges")
    print("  - dt shows temporal changes")
    print("  - Anomaly videos should have more motion than normal videos")


if __name__ == "__main__":
    main()
