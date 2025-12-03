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
    Each video is represented as a bag of features (segments, feature_dim).
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

        # Load and separate videos
        logger.info(f"Loading MIL features from: {feature_dir}")
        self.normal_videos, self.anomaly_videos = self._load_videos()

        logger.info(
            f"Loaded {len(self.normal_videos)} Normal videos, "
            f"{len(self.anomaly_videos)} Anomaly videos"
        )

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(self.normal_videos)
            random.shuffle(self.anomaly_videos)

    def _load_videos(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load all .npy feature files and separate into Normal/Anomaly.

        Returns:
            (normal_videos, anomaly_videos): Lists of numpy arrays
        """
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

                # Validate shape
                if features.ndim != 2:
                    logger.warning(
                        f"Skipping {file_path.name}: Expected 2D array, got shape {features.shape}"
                    )
                    continue

                # Check if video has correct number of segments
                if features.shape[0] != self.segments:
                    logger.warning(
                        f"Skipping {file_path.name}: Expected {self.segments} segments, "
                        f"got {features.shape[0]}"
                    )
                    continue

                # Separate by class
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
        Sample a balanced batch of Normal and Anomaly videos.

        Args:
            batch_size: Number of videos per class (total batch = 2 * batch_size)

        Returns:
            (norm_batch, anom_batch): Tensors of shape (batch_size, segments, feature_dim)
        """
        # Ensure we have enough videos
        if batch_size > len(self.normal_videos) or batch_size > len(
            self.anomaly_videos
        ):
            raise ValueError(
                f"Batch size {batch_size} exceeds available videos "
                f"(Normal={len(self.normal_videos)}, Anomaly={len(self.anomaly_videos)})"
            )

        # Sample random indices
        norm_indices = random.sample(range(len(self.normal_videos)), batch_size)
        anom_indices = random.sample(range(len(self.anomaly_videos)), batch_size)

        # Gather videos
        norm_batch = np.stack([self.normal_videos[i] for i in norm_indices])
        anom_batch = np.stack([self.anomaly_videos[i] for i in anom_indices])

        # Convert to tensors
        norm_batch = torch.from_numpy(norm_batch).float()
        anom_batch = torch.from_numpy(anom_batch).float()

        return norm_batch, anom_batch

    def __len__(self) -> int:
        """Return total number of videos."""
        return len(self.normal_videos) + len(self.anomaly_videos)

    def get_num_batches(self, batch_size: int) -> int:
        """
        Calculate number of batches per epoch.

        Limited by the smaller class (to maintain balance).
        """
        return min(len(self.normal_videos), len(self.anomaly_videos)) // batch_size
