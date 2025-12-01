"""Frame-level evaluation for temporal anomaly localization."""

import os
import re
import glob
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.amp import autocast
from tqdm import tqdm
from dataclasses import dataclass

from src.utils.evaluation_utils import extract_anomaly_scores, compute_auc_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameLevelMetrics:
    """Container for frame-level evaluation results."""

    frame_auc: float
    num_frames: int
    num_videos: int
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray

    def __str__(self):
        return (
            f"Frame-Level AUC: {self.frame_auc:.4f}\n"
            f"Total Frames: {self.num_frames:,}\n"
            f"Total Videos: {self.num_videos}"
        )


def parse_temporal_annotations(annotation_path):
    """
    Parse UCF-Crime temporal annotation file.

    Format: VideoName  Class  Start1  End1  Start2  End2
    Where -1 indicates no event.

    Args:
        annotation_path: Path to annotation file

    Returns:
        dict: {video_name: [(start1, end1), (start2, end2), ...]}
    """
    annotations = {}

    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                logger.warning(f"Malformed line: {line}")
                continue

            video_name = parts[0].replace(".mp4", "")
            class_name = parts[1]
            start1, end1, start2, end2 = map(int, parts[2:6])

            intervals = []
            if start1 != -1 and end1 != -1:
                intervals.append((start1, end1))
            if start2 != -1 and end2 != -1:
                intervals.append((start2, end2))

            annotations[video_name] = intervals

    logger.info(f"Parsed annotations for {len(annotations)} videos")
    return annotations


def get_test_videos(test_dir):
    """
    Scan test directory and group frames by video ID.

    Args:
        test_dir: Path to Test directory

    Returns:
        dict: {video_id: {'frames': [(idx, path), ...], 'class': class_name}}
    """
    videos = {}
    pattern = re.compile(r"(.+?)_x264_(\d+)\.png")

    class_folders = [
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    ]

    for class_name in class_folders:
        class_path = os.path.join(test_dir, class_name)
        image_files = sorted(glob.glob(os.path.join(class_path, "*.png")))

        for file_path in image_files:
            filename = os.path.basename(file_path)
            match = pattern.match(filename)

            if match:
                video_id = match.group(1) + "_x264"
                frame_idx = int(match.group(2))

                if video_id not in videos:
                    videos[video_id] = {"frames": [], "class": class_name}

                videos[video_id]["frames"].append((frame_idx, file_path))

    # Sort frames by index
    for video_id in videos:
        videos[video_id]["frames"].sort(key=lambda x: x[0])

    logger.info(f"Found {len(videos)} test videos")
    return videos


def create_ground_truth_mask(frame_indices, annotation_intervals):
    """
    Create binary mask for frames based on temporal annotations.

    Args:
        frame_indices: List of frame indices (e.g., [0, 10, 20, ...])
        annotation_intervals: List of (start, end) tuples

    Returns:
        np.ndarray: Binary mask (1 = anomaly, 0 = normal)
    """
    mask = np.zeros(len(frame_indices), dtype=np.int32)

    for start, end in annotation_intervals:
        for i, frame_idx in enumerate(frame_indices):
            if start <= frame_idx <= end:
                mask[i] = 1

    return mask


def sliding_window_inference(model, video_frames, clip_len, stride, transform, device):
    """
    Run model on video using sliding window.

    Args:
        model: Trained model
        video_frames: List of (frame_idx, frame_path) tuples
        clip_len: Frames per clip (16)
        stride: Stride for sliding window
        transform: RGBVideoTransform instance
        device: Device

    Returns:
        np.ndarray: Anomaly scores per frame
    """
    model.eval()
    num_frames = len(video_frames)

    # Initialize scores
    frame_scores = np.zeros(num_frames, dtype=np.float32)
    frame_counts = np.zeros(num_frames, dtype=np.int32)

    frame_paths = [path for _, path in video_frames]

    with torch.no_grad():
        for start_idx in range(0, num_frames - clip_len + 1, stride):
            end_idx = start_idx + clip_len

            # Load clip
            clip_frames = []
            for path in frame_paths[start_idx:end_idx]:
                frame = cv2.imread(path)
                if frame is None:
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames.append(frame)

            clip_frames = np.array(clip_frames)

            # Transform
            clip_tensor = transform(clip_frames)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)

            # Inference
            with autocast("cuda", enabled=True):
                outputs = model(clip_tensor)

            anomaly_score = extract_anomaly_scores(outputs)[0].cpu().item()

            # Assign score to all frames in clip
            for i in range(start_idx, end_idx):
                frame_scores[i] += anomaly_score
                frame_counts[i] += 1

    # Average overlapping predictions
    frame_counts = np.maximum(frame_counts, 1)
    frame_scores = frame_scores / frame_counts

    return frame_scores


def evaluate_frame_level(
    model,
    test_dir,
    annotation_path,
    transform,
    device,
    clip_len=16,
    stride=16,
    sigma=5,
) -> FrameLevelMetrics:
    """
    Perform frame-level evaluation on UCF-Crime test set.

    Args:
        model: Trained model
        test_dir: Path to Test directory
        annotation_path: Path to temporal annotations
        transform: RGBVideoTransform instance
        device: Device
        clip_len: Frames per clip
        stride: Sliding window stride
        sigma: Gaussian smoothing parameter

    Returns:
        FrameLevelMetrics dataclass
    """
    from sklearn.metrics import roc_curve

    logger.info("Starting frame-level evaluation...")

    # Parse annotations
    annotations = parse_temporal_annotations(annotation_path)
    videos = get_test_videos(test_dir)

    # Filter videos with annotations
    valid_videos = {k: v for k, v in videos.items() if k in annotations}
    logger.info(f"Processing {len(valid_videos)} videos with annotations")

    # Storage
    all_scores = []
    all_labels = []
    video_results = []  # For visualization

    # Process each video
    for video_id, video_data in tqdm(valid_videos.items(), desc="Processing videos"):
        frames = video_data["frames"]
        frame_indices = [idx for idx, _ in frames]

        # Ground truth
        intervals = annotations[video_id]
        gt_mask = create_ground_truth_mask(frame_indices, intervals)

        # Inference
        try:
            pred_scores = sliding_window_inference(
                model, frames, clip_len, stride, transform, device
            )
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            continue

        # Temporal smoothing
        if sigma > 0:
            pred_scores = gaussian_filter1d(pred_scores, sigma=sigma)

        # Compute per-video AUC
        video_auc = compute_auc_safe(gt_mask, pred_scores)
        video_results.append(
            (video_id, video_auc, pred_scores, gt_mask, frame_indices, intervals)
        )

        # Accumulate
        all_scores.extend(pred_scores)
        all_labels.extend(gt_mask)

    # Convert to arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute frame-level AUC
    frame_auc = compute_auc_safe(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    logger.info(f"Frame-Level AUC: {frame_auc:.4f}")

    metrics = FrameLevelMetrics(
        frame_auc=frame_auc,
        num_frames=len(all_labels),
        num_videos=len(valid_videos),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )

    return metrics, video_results
