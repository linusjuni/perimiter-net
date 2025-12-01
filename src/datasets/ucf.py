import os
import glob
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UCFCrimeDataset(Dataset):
    """UCF-Crime dataset for anomaly detection and classification."""

    # 14 Classes
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
        split="train",
        clip_len=16,
        stride=1,
        mode="binary",
        transform=None,
    ):
        """
        Args:
            root_dir (str): Path to data (e.g., '/work3/s225224/ucf-crime/data')
            split (str): 'train' or 'test'
            clip_len (int): Number of frames per clip
            stride (int): Frame jump size (1 = consecutive frames)
            mode (str): 'binary' (Normal vs Anomaly) or 'multiclass' (14 classes)
            transform (callable): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.mode = mode
        self.transform = transform

        # Map class names to integers
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # Replaces _load_annotations
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Parses directory structure instead of YAML.
        Returns list of dictionary: {'paths': [list of frame paths], 'label': int}
        """
        samples = []

        # Handle capitalization differences in folders (Train/Test)
        split_dir = "Train" if self.split.lower() == "train" else "Test"
        target_dir = os.path.join(self.root_dir, split_dir)

        print(f"Scanning {target_dir}...")

        # Iterate over each class folder (Abuse, Fighting, NormalVideos, etc.)
        for class_name in self.CLASSES:
            class_path = os.path.join(target_dir, class_name)
            if not os.path.exists(class_path):
                continue

            # Determine Label
            if self.mode == "binary":
                # Normal = 0, Everything else = 1
                label = 0 if class_name == "NormalVideos" else 1
            else:
                # Multiclass (0-13)
                label = self.class_to_idx[class_name]

            # 1. Get all images in this class folder
            image_files = sorted(glob.glob(os.path.join(class_path, "*.png")))

            # 2. Group frames by Video ID using Regex
            # Pattern: Fighting002_x264_1000.png -> ID: Fighting002_x264
            video_groups = {}
            pattern = re.compile(r"(.+?)_x264_(\d+)\.png")

            for file_path in image_files:
                filename = os.path.basename(file_path)
                match = pattern.match(filename)
                if match:
                    video_id = match.group(1)
                    # We store full path to avoid os.path.join later
                    if video_id not in video_groups:
                        video_groups[video_id] = []
                    video_groups[video_id].append(file_path)

            # 3. Create sliding window clips for each video
            for vid_id, frames in video_groups.items():
                # Ensure frames are sorted by name/number
                # (Glob is usually sorted, but this is a safety check)
                frames.sort(key=lambda x: int(re.search(r"_(\d+)\.png", x).group(1)))

                num_frames = len(frames)

                # If video is shorter than clip_len, skip or loop (here we skip)
                if num_frames < self.clip_len:
                    continue

                # Create clips: [0,16], [8,24], etc. based on stride
                for i in range(0, num_frames - self.clip_len + 1, self.stride):
                    clip_paths = frames[i : i + self.clip_len]
                    samples.append(
                        {
                            "paths": clip_paths,
                            "label": label,
                            "video_id": vid_id,  # Useful for debugging/evaluation
                        }
                    )

        print(f"Loaded {len(samples)} clips for {self.split} split.")
        return samples

    def _load_video_clip(self, frame_paths):
        """
        Load video clip from a list of image paths.
        Replaces cv2.VideoCapture logic.
        """
        frames = []
        for path in frame_paths:
            # Read image
            frame = cv2.imread(path)

            # Safety check if image is corrupt or missing
            if frame is None:
                # Fallback: create black frame or copy previous
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    # Should rarely happen
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Convert BGR (OpenCV) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        # Ensure we have exactly clip_len frames
        # (Though logic in _load_samples guarantees this, it's good safety)
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        return np.array(frames)

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return (video_clip, label) tuple."""
        # Get sample metadata
        sample = self.samples[idx]
        frame_paths = sample["paths"]
        label = sample["label"]

        # Load actual pixel data
        frames = self._load_video_clip(frame_paths)

        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        else:
            # Default transform if none provided: (T, H, W, C) -> (C, T, H, W)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        return frames, label
