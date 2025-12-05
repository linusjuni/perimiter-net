import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import random
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MILDataLoader:
    """Multiple Instance Learning DataLoader for UCF-Crime."""

    def __init__(
        self,
        feature_dir: str,
        segments: int = 32,
        shuffle: bool = True,
        split: str = "train",
        val_split: float = 0.2,
        random_seed: int = 69,
    ):
        """Initialize MILDataLoader."""
        self.feature_dir = Path(feature_dir)
        self.segments = segments
        self.shuffle = shuffle
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed

        # Load and separate videos
        logger.info(f"Loading MIL features from: {feature_dir} (split={split})")
        self.normal_videos, self.anomaly_videos = self._load_videos()

        logger.info(
            f"[{split.upper()}] Loaded {len(self.normal_videos)} Normal videos, "
            f"{len(self.anomaly_videos)} Anomaly videos"
        )

        # Shuffle if requested (after splitting, so it doesn't affect reproducibility)
        if self.shuffle:
            random.shuffle(self.normal_videos)
            random.shuffle(self.anomaly_videos)

    def _interpolate(self, features: np.ndarray) -> np.ndarray:
        """Interpolate features to fixed segments."""
        T, D = features.shape

        if T == self.segments:
            return features

        chunks = np.array_split(features, self.segments, axis=0)
        interpolated = np.zeros((self.segments, D), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
                # interpolated[i] = np.max(chunk, axis=0)
            else:
                interpolated[i] = np.zeros(D)

        return interpolated

    def _load_videos(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load and split videos."""
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")

        all_files = list(self.feature_dir.glob("*.npy"))

        if len(all_files) == 0:
            raise ValueError(f"No .npy files found in {self.feature_dir}")

        # Separate by class first
        normal_files = []
        anomaly_files = []

        for file_path in all_files:
            if "Normal" in file_path.name:
                normal_files.append(file_path)
            else:
                anomaly_files.append(file_path)

        # Deterministic split (sort by name for reproducibility)
        normal_files = sorted(normal_files, key=lambda x: x.name)
        anomaly_files = sorted(anomaly_files, key=lambda x: x.name)

        # Set seed for reproducible split
        rng = np.random.RandomState(self.random_seed)

        # Split indices
        n_norm_val = int(len(normal_files) * self.val_split)
        n_anom_val = int(len(anomaly_files) * self.val_split)

        # Shuffle indices (but deterministically)
        norm_indices = np.arange(len(normal_files))
        anom_indices = np.arange(len(anomaly_files))
        rng.shuffle(norm_indices)
        rng.shuffle(anom_indices)

        # Select files based on split
        if self.split == "val":
            normal_files = [normal_files[i] for i in norm_indices[:n_norm_val]]
            anomaly_files = [anomaly_files[i] for i in anom_indices[:n_anom_val]]
        else:  # train
            normal_files = [normal_files[i] for i in norm_indices[n_norm_val:]]
            anomaly_files = [anomaly_files[i] for i in anom_indices[n_anom_val:]]

        # Load features
        normal_videos = []
        anomaly_videos = []

        for file_path in normal_files:
            try:
                features = np.load(file_path)
                if features.ndim != 2 or features.shape[0] == 0:
                    continue
                features = self._interpolate(features)
                normal_videos.append(features)
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue

        for file_path in anomaly_files:
            try:
                features = np.load(file_path)
                if features.ndim != 2 or features.shape[0] == 0:
                    continue
                features = self._interpolate(features)
                anomaly_videos.append(features)
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue

        if len(normal_videos) == 0 or len(anomaly_videos) == 0:
            raise ValueError(
                f"Invalid dataset: Normal={len(normal_videos)}, Anomaly={len(anomaly_videos)}"
            )

        return normal_videos, anomaly_videos

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a balanced batch of videos."""
        n_sample = min(batch_size, len(self.normal_videos))
        a_sample = min(batch_size, len(self.anomaly_videos))

        if n_sample < batch_size:
            norm_indices = np.random.choice(
                len(self.normal_videos), batch_size, replace=True
            )
        else:
            norm_indices = random.sample(range(len(self.normal_videos)), batch_size)

        if a_sample < batch_size:
            anom_indices = np.random.choice(
                len(self.anomaly_videos), batch_size, replace=True
            )
        else:
            anom_indices = random.sample(range(len(self.anomaly_videos)), batch_size)

        norm_batch = np.stack([self.normal_videos[i] for i in norm_indices])
        anom_batch = np.stack([self.anomaly_videos[i] for i in anom_indices])

        return torch.from_numpy(norm_batch).float(), torch.from_numpy(
            anom_batch
        ).float()

    def __len__(self) -> int:
        """Total number of videos."""
        return len(self.normal_videos) + len(self.anomaly_videos)

    def get_num_batches(self, batch_size: int) -> int:
        """Number of batches per epoch."""
        return min(len(self.normal_videos), len(self.anomaly_videos)) // batch_size
