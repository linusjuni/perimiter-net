"""Create train/val/test splits for VIRAT dataset using only available videos."""

from pathlib import Path
import random

# Paths
video_dir = Path("/work3/s225224/perimeter-net/data/videos")
split_dir = Path("/work3/s225224/perimeter-net/data/splits")
annot_dir = Path("/work3/s225224/perimeter-net/data/raw/viratannotations")

split_dir.mkdir(parents=True, exist_ok=True)

# Get videos that actually exist
existing_videos = set(f.stem for f in video_dir.glob("*.mp4"))
print(f"Found {len(existing_videos)} videos in {video_dir}")

# Get annotated videos from train folder
train_videos = [
    f.stem.replace(".activities", "")
    for f in (annot_dir / "train").glob("*.activities.yml")
]

# Get annotated videos from validate folder
validate_videos = [
    f.stem.replace(".activities", "")
    for f in (annot_dir / "validate").glob("*.activities.yml")
]

# Filter to only videos that exist
train_videos = [v for v in train_videos if v in existing_videos]
validate_videos = [v for v in validate_videos if v in existing_videos]

print(f"\nAnnotated videos: {len(train_videos)} train, {len(validate_videos)} validate")

# Split validate into val and test
random.seed(42)
validate_videos_sorted = sorted(validate_videos)
random.shuffle(validate_videos_sorted)

val_split_idx = int(len(validate_videos_sorted) * 0.55)
val_videos = validate_videos_sorted[:val_split_idx]
test_videos = validate_videos_sorted[val_split_idx:]

# Write splits
with open(split_dir / "train.txt", "w") as f:
    f.write("\n".join(sorted(train_videos)) + "\n")

with open(split_dir / "val.txt", "w") as f:
    f.write("\n".join(sorted(val_videos)) + "\n")

with open(split_dir / "test.txt", "w") as f:
    f.write("\n".join(sorted(test_videos)) + "\n")

print("\n✓ Created splits:")
print(f"  Train: {len(train_videos)} videos")
print(f"  Val:   {len(val_videos)} videos")
print(f"  Test:  {len(test_videos)} videos")
print(f"  Total: {len(train_videos) + len(val_videos) + len(test_videos)} videos")
