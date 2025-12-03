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
    Supports deterministic train/val splitting.
    """

    def __init__(
        self,
        feature_dir: str,
        segments: int = 32,
        shuffle: bool = True,
        split: str = "train",
        val_split: float = 0.2,
        random_seed: int = 69,
    ):
        """
        Args:
            feature_dir: Path to directory containing .npy feature files
            segments: Number of segments per video (default: 32)
            shuffle: Whether to shuffle video lists (default: True)
            split: 'train' or 'val'
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splits (default: 42)
        """
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
        """Compress variable length video features into fixed 32 segments."""
        T, D = features.shape

        if T == self.segments:
            return features

        chunks = np.array_split(features, self.segments, axis=0)
        interpolated = np.zeros((self.segments, D), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
            else:
                interpolated[i] = np.zeros(D)

        return interpolated

    def _load_videos(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load videos and split into train/val deterministically."""
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
        """Sample a balanced batch of Normal and Anomaly videos."""
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
        """Return total number of videos."""
        return len(self.normal_videos) + len(self.anomaly_videos)

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate number of batches per epoch."""
        return min(len(self.normal_videos), len(self.anomaly_videos)) // batch_size


class TwoStreamMILDataLoader:
    """
    Two-Stream MIL DataLoader for UCF-Crime.

    Loads paired RGB and Motion features for the same videos.
    Maintains separate lists of Normal and Anomaly videos for balanced sampling.
    """

    def __init__(
        self,
        rgb_feature_dir: str,
        motion_feature_dir: str,
        segments: int = 32,
        shuffle: bool = True,
        split: str = "train",
        val_split: float = 0.2,
        random_seed: int = 69,
    ):
        """
        Args:
            rgb_feature_dir: Path to directory containing RGB .npy feature files
            motion_feature_dir: Path to directory containing Motion .npy feature files
            segments: Number of segments per video (default: 32)
            shuffle: Whether to shuffle video lists (default: True)
            split: 'train' or 'val'
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splits (default: 69)
        """
        self.rgb_feature_dir = Path(rgb_feature_dir)
        self.motion_feature_dir = Path(motion_feature_dir)
        self.segments = segments
        self.shuffle = shuffle
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed

        # Load and separate videos (paired)
        logger.info(f"Loading Two-Stream MIL features (split={split})")
        logger.info(f"  RGB: {rgb_feature_dir}")
        logger.info(f"  Motion: {motion_feature_dir}")

        self.normal_videos, self.anomaly_videos = self._load_paired_videos()

        logger.info(
            f"[{split.upper()}] Loaded {len(self.normal_videos)} Normal videos, "
            f"{len(self.anomaly_videos)} Anomaly videos (paired)"
        )

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(self.normal_videos)
            random.shuffle(self.anomaly_videos)

    def _interpolate(self, features: np.ndarray) -> np.ndarray:
        """Compress variable length video features into fixed segments."""
        T, D = features.shape

        if T == self.segments:
            return features

        chunks = np.array_split(features, self.segments, axis=0)
        interpolated = np.zeros((self.segments, D), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
            else:
                interpolated[i] = np.zeros(D)

        return interpolated

    def _load_paired_videos(
        self,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Load paired RGB and Motion features.

        Returns:
            Tuple of (normal_videos, anomaly_videos) where each video is a dict:
            {'rgb': np.ndarray, 'motion': np.ndarray, 'name': str}
        """
        if not self.rgb_feature_dir.exists():
            raise FileNotFoundError(
                f"RGB feature directory not found: {self.rgb_feature_dir}"
            )
        if not self.motion_feature_dir.exists():
            raise FileNotFoundError(
                f"Motion feature directory not found: {self.motion_feature_dir}"
            )

        # Get all RGB files and find matching Motion files
        rgb_files = {f.stem: f for f in self.rgb_feature_dir.glob("*.npy")}
        motion_files = {f.stem: f for f in self.motion_feature_dir.glob("*.npy")}

        # Find common videos (must have both RGB and Motion)
        common_names = set(rgb_files.keys()) & set(motion_files.keys())

        if len(common_names) == 0:
            raise ValueError(
                f"No matching video pairs found between RGB and Motion directories. "
                f"RGB has {len(rgb_files)} files, Motion has {len(motion_files)} files."
            )

        # Report any mismatches
        rgb_only = set(rgb_files.keys()) - common_names
        motion_only = set(motion_files.keys()) - common_names

        if rgb_only:
            logger.warning(f"Found {len(rgb_only)} RGB-only videos (no Motion pair)")
        if motion_only:
            logger.warning(f"Found {len(motion_only)} Motion-only videos (no RGB pair)")

        logger.info(f"Found {len(common_names)} paired videos")

        # Separate by class
        normal_names = sorted([n for n in common_names if "Normal" in n])
        anomaly_names = sorted([n for n in common_names if "Normal" not in n])

        # Deterministic split
        rng = np.random.RandomState(self.random_seed)

        n_norm_val = int(len(normal_names) * self.val_split)
        n_anom_val = int(len(anomaly_names) * self.val_split)

        norm_indices = np.arange(len(normal_names))
        anom_indices = np.arange(len(anomaly_names))
        rng.shuffle(norm_indices)
        rng.shuffle(anom_indices)

        # Select based on split
        if self.split == "val":
            normal_names = [normal_names[i] for i in norm_indices[:n_norm_val]]
            anomaly_names = [anomaly_names[i] for i in anom_indices[:n_anom_val]]
        else:  # train
            normal_names = [normal_names[i] for i in norm_indices[n_norm_val:]]
            anomaly_names = [anomaly_names[i] for i in anom_indices[n_anom_val:]]

        # Load paired features
        normal_videos = self._load_video_pairs(normal_names, rgb_files, motion_files)
        anomaly_videos = self._load_video_pairs(anomaly_names, rgb_files, motion_files)

        if len(normal_videos) == 0 or len(anomaly_videos) == 0:
            raise ValueError(
                f"Invalid dataset: Normal={len(normal_videos)}, Anomaly={len(anomaly_videos)}"
            )

        return normal_videos, anomaly_videos

    def _load_video_pairs(
        self,
        video_names: List[str],
        rgb_files: dict,
        motion_files: dict,
    ) -> List[dict]:
        """Load RGB and Motion features for a list of video names."""
        videos = []

        for name in video_names:
            try:
                rgb_features = np.load(rgb_files[name])
                motion_features = np.load(motion_files[name])

                # Validate shapes
                if rgb_features.ndim != 2 or rgb_features.shape[0] == 0:
                    logger.warning(f"Invalid RGB features for {name}, skipping")
                    continue
                if motion_features.ndim != 2 or motion_features.shape[0] == 0:
                    logger.warning(f"Invalid Motion features for {name}, skipping")
                    continue

                # Interpolate to fixed segments
                rgb_features = self._interpolate(rgb_features)
                motion_features = self._interpolate(motion_features)

                videos.append(
                    {
                        "rgb": rgb_features,
                        "motion": motion_features,
                        "name": name,
                    }
                )

            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                continue

        return videos

    def get_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a balanced batch of Normal and Anomaly videos.

        Returns:
            Tuple of (normal_rgb, normal_motion, anomaly_rgb, anomaly_motion)
            Each tensor has shape (batch_size, segments, feature_dim)
        """
        # Sample indices
        if len(self.normal_videos) < batch_size:
            norm_indices = np.random.choice(
                len(self.normal_videos), batch_size, replace=True
            )
        else:
            norm_indices = random.sample(range(len(self.normal_videos)), batch_size)

        if len(self.anomaly_videos) < batch_size:
            anom_indices = np.random.choice(
                len(self.anomaly_videos), batch_size, replace=True
            )
        else:
            anom_indices = random.sample(range(len(self.anomaly_videos)), batch_size)

        # Stack features
        norm_rgb = np.stack([self.normal_videos[i]["rgb"] for i in norm_indices])
        norm_motion = np.stack([self.normal_videos[i]["motion"] for i in norm_indices])
        anom_rgb = np.stack([self.anomaly_videos[i]["rgb"] for i in anom_indices])
        anom_motion = np.stack([self.anomaly_videos[i]["motion"] for i in anom_indices])

        return (
            torch.from_numpy(norm_rgb).float(),
            torch.from_numpy(norm_motion).float(),
            torch.from_numpy(anom_rgb).float(),
            torch.from_numpy(anom_motion).float(),
        )

    def __len__(self) -> int:
        """Return total number of videos."""
        return len(self.normal_videos) + len(self.anomaly_videos)

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate number of batches per epoch."""
        return min(len(self.normal_videos), len(self.anomaly_videos)) // batch_size