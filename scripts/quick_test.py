"""Test RGBVideoTransform integration with VIRATDataset."""

import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)

sys.path.append("/work3/s225224/perimeter-net")

from src.datasets.virat import VIRATDataset
from src.datasets.transforms import RGBVideoTransform

logger.info("Starting quick test for VIRATDataset with RGBVideoTransform")
logger.warning("Ensure that the VIRAT dataset is available at the specified path.")

print("=" * 60)
print("Testing VIRATDataset with RGBVideoTransform")
print("=" * 60)

# Test 1: Training transform
print("\n[Test 1] Training transform (with augmentations)")
train_transform = RGBVideoTransform(mode="train", crop_size=112, resize_size=128)
train_dataset = VIRATDataset(
    root_dir="/work3/s225224/perimeter-net/data",
    split="train",
    clip_len=16,
    transform=train_transform,
)

print(f"✓ Dataset loaded: {len(train_dataset)} samples")

# Load one sample
video_clip, label = train_dataset[0]
print(f"✓ Video clip shape: {video_clip.shape}")
print(f"✓ Label: {label} ({train_dataset.ACTIVITIES[label]})")
print(f"✓ Clip dtype: {video_clip.dtype}")
print(f"✓ Value range: [{video_clip.min():.3f}, {video_clip.max():.3f}]")

# Expected: (3, 16, 112, 112)
assert video_clip.shape == (3, 16, 112, 112), f"Wrong shape: {video_clip.shape}"
print("✓ Shape is correct: (3, 16, 112, 112)")

# Test 2: Validation transform
print("\n[Test 2] Validation transform (no augmentations)")
val_transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)
val_dataset = VIRATDataset(
    root_dir="/work3/s225224/perimeter-net/data",
    split="val",
    clip_len=16,
    transform=val_transform,
)

print(f"✓ Dataset loaded: {len(val_dataset)} samples")

video_clip, label = val_dataset[0]
print(f"✓ Video clip shape: {video_clip.shape}")
print(f"✓ Label: {label} ({val_dataset.ACTIVITIES[label]})")

# Test 3: No transform (fallback)
print("\n[Test 3] No transform (fallback to manual conversion)")
no_transform_dataset = VIRATDataset(
    root_dir="/work3/s225224/perimeter-net/data",
    split="val",
    clip_len=16,
    transform=None,
)

video_clip, label = no_transform_dataset[0]
print(f"✓ Video clip shape: {video_clip.shape}")
print(f"✓ Value range: [{video_clip.min():.3f}, {video_clip.max():.3f}]")

# Test 4: Load multiple samples (check for crashes)
print("\n[Test 4] Loading multiple samples")
for i in range(min(5, len(train_dataset))):
    video_clip, label = train_dataset[i]
    assert video_clip.shape == (3, 16, 112, 112)
print(f"✓ Successfully loaded 5 samples")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
