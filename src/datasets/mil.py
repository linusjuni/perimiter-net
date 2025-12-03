import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import random
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MILDataLoader:
    """
    Multiple Instance Learning DataLoader for UCF-Crime.
    Maintains separate lists of Normal and Anomaly videos for balanced sampling.
    """

    def __init__(
        self,
        feature_dir: str,
        segments: int = 32,
        shuffle: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.segments = segments
        self.shuffle = shuffle

        logger.info(f"Loading MIL features from: {feature_dir}")
        self.normal_videos, self.anomaly_videos = self._load_videos()

        logger.info(
            f"Loaded {len(self.normal_videos)} Normal videos, "
            f"{len(self.anomaly_videos)} Anomaly videos"
        )

        if self.shuffle:
            random.shuffle(self.normal_videos)
            random.shuffle(self.anomaly_videos)

    def _interpolate(self, features: np.ndarray) -> np.ndarray:
        """
        Compress variable length video features into fixed 32 segments.
        """
        T, D = features.shape

        # If already correct size, return
        if T == self.segments:
            return features

        # Split into 32 chunks and average them
        # np.array_split handles uneven splits automatically
        chunks = np.array_split(features, self.segments, axis=0)

        interpolated = np.zeros((self.segments, D), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
            else:
                # Handle edge case for extremely short videos (rare)
                interpolated[i] = np.zeros(D)

        return interpolated

    def _load_videos(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")

        all_files = list(self.feature_dir.glob("*.npy"))
        if len(all_files) == 0:
            raise ValueError(f"No .npy files found in {self.feature_dir}")

        normal_videos = []
        anomaly_videos = []

        for file_path in all_files:
            try:
                features = np.load(file_path)

                # Validate dimensions
                if features.ndim != 2:
                    continue

                # Skip empty files
                if features.shape[0] == 0:
                    continue

                # --- CRITICAL CHANGE: INTERPOLATE HERE ---
                # Instead of skipping, we force it to 32 segments
                features = self._interpolate(features)
                # -----------------------------------------

                if "Normal" in file_path.name:
                    normal_videos.append(features)
                else:
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
        """
        Returns: (norm_batch, anom_batch) tensors
        """
        # Ensure we don't request more than we have
        # If batch is too big, just grab as many as possible
        n_sample = min(batch_size, len(self.normal_videos))
        a_sample = min(batch_size, len(self.anomaly_videos))

        if n_sample < batch_size:
            # Option: Sample with replacement if dataset is tiny
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

    def get_num_batches(self, batch_size: int) -> int:
        return min(len(self.normal_videos), len(self.anomaly_videos)) // batch_size
