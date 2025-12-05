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
    """Visualize motion overlay."""
    clip_np = clip_tensor.cpu().numpy()
    C, T, H, W = clip_np.shape
    mid_frame = T // 2

    dx = clip_np[0, mid_frame]
    dy = clip_np[1, mid_frame]
    dt = clip_np[2, mid_frame]

    orig_frame = cv2.imread(frame_paths[mid_frame])
    orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_frame.shape[:2]

    magnitude = np.sqrt(dx**2 + dy**2 + dt**2)
    magnitude_norm = (magnitude - magnitude.min()) / (
        magnitude.max() - magnitude.min() + 1e-8
    )

    magnitude_fullres = cv2.resize(
        magnitude_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    brightened_frame = np.clip(orig_frame * 1.2, 0, 255).astype(np.uint8)

    dpi = 100
    fig = plt.figure(figsize=(orig_w / dpi, orig_h / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(brightened_frame)
    ax.imshow(magnitude_fullres, cmap="jet", alpha=0.6)

    return fig


def print_statistics(clip_tensor, label, class_name):
    """Print statistics about motion features."""
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

        non_zero = np.count_nonzero(np.abs(channel_data) > 0.01)
        total = channel_data.size
        print(f"  Motion Pixels: {non_zero}/{total} ({100 * non_zero / total:.1f}%)")


def main():
    """Run Sobel motion transform smoke test."""
    root_dir = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Frames"
    clip_len = 16
    num_samples = 5

    print("=" * 60)
    print("SOBEL MOTION TRANSFORM SMOKE TEST")
    print("=" * 60)

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

    print(f"\nLoading {num_samples} samples...")

    anomaly_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 1]
    normal_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]

    test_indices = []
    anomaly_step = max(1, len(anomaly_indices) // (num_samples // 2 + 1))
    normal_step = max(1, len(normal_indices) // (num_samples // 2 + 1))

    for i in range(num_samples):
        if i % 2 == 0 and len(anomaly_indices) > 0:
            idx = (i // 2) * anomaly_step
            if idx < len(anomaly_indices):
                test_indices.append(anomaly_indices[idx])
        else:
            idx = (i // 2) * normal_step
            if idx < len(normal_indices):
                test_indices.append(normal_indices[idx])

    for sample_num, idx in enumerate(test_indices):
        print(f"\n--- Sample {sample_num + 1}/{len(test_indices)} ---")

        frame_paths = dataset.samples[idx]["paths"]
        clip, label = dataset[idx]

        class_folder = Path(frame_paths[0]).parent.name

        print_statistics(clip, label, class_folder)

        fig = visualize_motion_clip(
            clip, frame_paths, sample_idx=sample_num + 1, class_name=class_folder
        )

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
