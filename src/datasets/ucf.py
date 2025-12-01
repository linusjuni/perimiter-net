import os
import glob
import re
import cv2
import numpy as np
import hashlib
import pickle
from torch.utils.data import Dataset
from src.datasets.transforms import RGBVideoTransform


class UCFCrimeDataset(Dataset):
    """UCF-Crime dataset with deterministic Train/Val splitting."""

    CLASSES = [
        "NormalVideos",
        "Abuse",
        "Arrest",
        "Arson",
        "Assault",
        "Burglary",
        "Explosion",
        "Fighting",
        "RoadAccidents",
        "Robbery",
        "Shooting",
        "Shoplifting",
        "Stealing",
        "Vandalism",
    ]

    def __init__(
        self,
        root_dir,
        split="train",  # 'train', 'val', or 'test'
        clip_len=16,
        stride=1,
        mode="binary",
        transform=None,
        val_ratio=0.25,  # 25% of Training folder -> ~20% of total data
    ):
        self.root_dir = root_dir
        self.split = split.lower()
        self.clip_len = clip_len
        self.stride = stride
        self.mode = mode
        self.val_ratio = val_ratio

        # Map class names to integers
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # Load samples with caching
        self.samples = self._load_samples_with_cache()

        # Default Transform
        if transform is None:
            self.transform = RGBVideoTransform(
                mode="train" if self.split == "train" else "val",
                crop_size=112,
                resize_size=128,
            )
        else:
            self.transform = transform

    def _is_val_video(self, video_name):
        """
        Deterministically decide if a video belongs to validation set.
        Uses MD5 hash of the video name to ensure consistency across runs.
        """
        hash_object = hashlib.md5(video_name.encode())
        # Use first 8 chars of hex digest for integer conversion
        hash_int = int(hash_object.hexdigest()[:8], 16)
        # Normalize to [0, 1]
        normalized_hash = hash_int / 0xFFFFFFFF
        return normalized_hash < self.val_ratio

    def _load_samples_with_cache(self):
        """Wrapper to handle caching of the dataset index."""
        cache_name = (
            f"ucf_index_{self.split}_"
            f"len{self.clip_len}_"
            f"stride{self.stride}_"
            f"mode{self.mode}_"
            f"rat{self.val_ratio}.pkl"
        )

        cache_path = os.path.join(self.root_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Loading cached dataset index from {cache_path}...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"Indexing dataset for split '{self.split}' (this may take a while)...")
        samples = self._load_samples()

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
            print(f"Saved index to {cache_path}")

        return samples

    def _load_samples(self):
        samples = []

        # 1. Determine Source Folder
        # 'train' and 'val' splits both come from the 'Train' folder on disk
        # 'test' split comes from the 'Test' folder on disk
        is_train_disk_source = self.split in ["train", "val"]
        source_folder_name = "Train" if is_train_disk_source else "Test"
        target_dir = os.path.join(self.root_dir, source_folder_name)

        if not os.path.exists(target_dir):
            raise ValueError(f"Directory not found: {target_dir}")

        print(f"Scanning {target_dir}...")

        for class_name in self.CLASSES:
            class_path = os.path.join(target_dir, class_name)
            if not os.path.exists(class_path):
                continue

            # Determine Label
            if self.mode == "binary":
                label = 0 if class_name == "NormalVideos" else 1
            else:
                label = self.class_to_idx[class_name]

            # Get all images
            image_files = sorted(glob.glob(os.path.join(class_path, "*.png")))

            # Group by Video ID
            video_groups = {}
            # Regex to extract Video ID (e.g., "Abuse001_x264" from "Abuse001_x264_100.png")
            pattern = re.compile(r"(.+?)_\d+\.png")

            for file_path in image_files:
                filename = os.path.basename(file_path)
                # Simple split usually works better than complex regex for this dataset
                # Format is usually: VideoName_FrameNum.png
                # But UCF-Crime extraction often results in: Name_x264_Num.png
                parts = filename.rsplit("_", 1)
                if len(parts) == 2:
                    vid_id = parts[0]
                    if vid_id not in video_groups:
                        video_groups[vid_id] = []
                    video_groups[vid_id].append(file_path)

            # Process each video
            for vid_id, frames in video_groups.items():
                # --- SPLIT LOGIC ---
                if is_train_disk_source:
                    is_val = self._is_val_video(vid_id)

                    # If we want 'train' split, skip validation videos
                    if self.split == "train" and is_val:
                        continue

                    # If we want 'val' split, skip training videos
                    if self.split == "val" and not is_val:
                        continue
                # -------------------

                # Sort frames numerically
                frames.sort(key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0]))

                num_frames = len(frames)
                if num_frames < self.clip_len:
                    continue

                # Create clips
                for i in range(0, num_frames - self.clip_len + 1, self.stride):
                    clip_paths = frames[i : i + self.clip_len]
                    samples.append({"paths": clip_paths, "label": label})

        print(f"Loaded {len(samples)} clips for {self.split} split.")
        return samples

    def _load_video_clip(self, frame_paths):
        frames = []
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                # Create black frame if missing
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return np.array(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self._load_video_clip(sample["paths"])
        if self.transform:
            frames = self.transform(frames)
        return frames, sample["label"]
