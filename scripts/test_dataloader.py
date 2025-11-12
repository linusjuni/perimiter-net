"""Test that VIRATDataset loads correctly."""
from src.datasets.virat import VIRATDataset

print("Loading VIRATDataset...")

# Test loading train split
dataset = VIRATDataset(
    root_dir="/work3/s225224/perimeter-net/data",
    split="train",
    clip_len=16
)

print(f"Dataset loaded: {len(dataset)} samples")
print(f"Activities: {dataset.ACTIVITIES}")

# Test loading one sample
if len(dataset) > 0:
    video_clip, label = dataset[0]
    print(f"Video clip shape: {video_clip.shape}")
    print(f"Label: {label} ({dataset.ACTIVITIES[label]})")
    print("\nDataloader test passed!")
else:
    print("No samples found!")