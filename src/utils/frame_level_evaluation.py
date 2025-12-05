"""Frame-level evaluation for anomaly localization."""

import os
import re
import glob
import cv2
import pickle
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from torch.amp import autocast
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from src.utils.evaluation_utils import (
    compute_auc_safe,
    compute_youdens_j,
    extract_anomaly_scores,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameLevelMetrics:
    """Frame-level evaluation results container."""

    frame_auc: float
    num_frames: int
    num_videos: int
    decision_threshold: float
    youden_j: float
    youden_threshold: float

    # Raw counts
    tp: int
    tn: int
    fp: int
    fn: int

    # Per-class metrics
    anomaly_precision: float
    anomaly_recall: float
    anomaly_f1: float
    normal_precision: float
    normal_recall: float
    normal_f1: float

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def fnr(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0

    @property
    def tpr(self) -> float:
        return self.anomaly_recall

    def __str__(self):
        return (
            f"Frame-Level AUC: {self.frame_auc:.4f}\n"
            f"   >> Threshold: {self.decision_threshold:.3f} (Youden J={self.youden_j:.3f} @ {self.youden_threshold:.3f}) | Frames: {self.num_frames:,} | Videos: {self.num_videos}\n"
            f"   >> Anomaly: P={self.anomaly_precision:.3f} R={self.anomaly_recall:.3f} F1={self.anomaly_f1:.3f}\n"
            f"   >> Normal:  P={self.normal_precision:.3f} R={self.normal_recall:.3f} F1={self.normal_f1:.3f}\n"
            f"   >> Counts:  TP={self.tp} FN={self.fn} | TN={self.tn} FP={self.fp} | FPR={self.fpr:.4f} FNR={self.fnr:.4f}"
        )


@dataclass
class FrameLevelCurves:
    """Curve data for downstream plotting."""

    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    pr_thresholds: np.ndarray


def parse_temporal_annotations(annotation_path):
    """Parse UCF-Crime temporal annotation file."""
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
    """Scan test directory and group frames by video ID."""
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
    """Create binary mask for frames based on annotations."""
    mask = np.zeros(len(frame_indices), dtype=np.int32)

    for start, end in annotation_intervals:
        for i, frame_idx in enumerate(frame_indices):
            if start <= frame_idx <= end:
                mask[i] = 1

    return mask


def sliding_window_inference(model, video_frames, clip_len, stride, transform, device):
    """Run model on video using sliding window."""
    model.eval()
    num_frames = len(video_frames)

    # Initialize scores
    frame_scores = np.zeros(num_frames, dtype=np.float32)
    frame_counts = np.zeros(num_frames, dtype=np.int32)

    frame_paths = [path for _, path in video_frames]

    # Pre-allocate clip buffer (reuse across iterations)
    clip_buffer = np.zeros((clip_len, 64, 64, 3), dtype=np.uint8)

    with torch.no_grad():
        for start_idx in range(0, num_frames - clip_len + 1, stride):
            end_idx = start_idx + clip_len

            # Load clip into pre-allocated buffer
            for i, path in enumerate(frame_paths[start_idx:end_idx]):
                frame = cv2.imread(path)
                if frame is None:
                    clip_buffer[i] = 0  # Black frame
                else:
                    clip_buffer[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform (creates new tensor, but we reuse clip_buffer)
            clip_tensor = transform(clip_buffer)
            clip_tensor = clip_tensor.unsqueeze(0).to(device, non_blocking=True)

            # Inference
            with autocast("cuda", enabled=True):
                outputs = model(clip_tensor)

            anomaly_score = extract_anomaly_scores(outputs)[0].cpu().item()

            # Assign score to all frames in clip
            frame_scores[start_idx:end_idx] += anomaly_score
            frame_counts[start_idx:end_idx] += 1

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
    stride=8,
    sigma=5,
    decision_threshold: Optional[float] = None,
) -> Tuple[FrameLevelMetrics, FrameLevelCurves, list, np.ndarray, np.ndarray]:
    """Perform frame-level evaluation on UCF-Crime test set."""
    logger.info("Starting frame-level evaluation...")

    # Parse annotations
    annotations = parse_temporal_annotations(annotation_path)
    videos = get_test_videos(test_dir)

    # Handle videos not in annotation file (treat as normal)
    for video_id in videos:
        if video_id not in annotations:
            logger.warning(f"Video {video_id} not in annotations - treating as Normal")
            annotations[video_id] = []  # Empty intervals = all normal

    # Now all videos are valid
    valid_videos = videos
    logger.info(f"Processing {len(valid_videos)} videos (with annotations or defaults)")

    # Storage
    all_scores = []
    all_labels = []
    video_results = []  # For visualization

    # Process each video
    for video_id, video_data in tqdm(valid_videos.items(), desc="Processing videos"):
        frames = video_data["frames"]
        frame_indices = [idx for idx, _ in frames]

        # Ground truth (will be all zeros if intervals is empty)
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

        # Compute per-video AUC only if both classes are present
        if len(np.unique(gt_mask)) > 1:
            video_auc = compute_auc_safe(gt_mask, pred_scores)
            video_results.append(
                (video_id, video_auc, pred_scores, gt_mask, frame_indices, intervals)
            )
        else:
            # Still keep normal-only videos for visualization purposes
            logger.debug(
                f"Skipping per-video AUC for {video_id} (single class: {'Normal' if gt_mask.sum() == 0 else 'Anomaly'})"
            )

        # Accumulate for global frame-level AUC (includes normal-only videos)
        all_scores.extend(pred_scores)
        all_labels.extend(gt_mask)

    # Convert to arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute frame-level AUC
    frame_auc = compute_auc_safe(all_labels, all_scores)
    logger.info(f"Frame-Level AUC: {frame_auc:.4f}")

    # Curves for downstream plotting (ROC + PR)
    if len(np.unique(all_labels)) > 1:
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(all_labels, all_scores)
        youden_threshold, youden_j = compute_youdens_j(roc_fpr, roc_tpr, roc_thresholds)
    else:
        logger.warning(
            "Only one class present in labels; ROC curve and Youden's J cannot be computed."
        )
        roc_fpr = np.array([])
        roc_tpr = np.array([])
        roc_thresholds = np.array([])
        youden_threshold = float("nan")
        youden_j = float("nan")

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        all_labels, all_scores
    )

    # Select decision threshold (Youden's J if none provided)
    threshold_to_use = (
        youden_threshold if decision_threshold is None else float(decision_threshold)
    )

    if np.isnan(threshold_to_use):
        threshold_to_use = 0.5
        logger.warning("Decision threshold undefined; falling back to 0.5.")
    elif decision_threshold is None:
        logger.info(
            f"Using Youden's J threshold: {threshold_to_use:.4f} (J={youden_j:.4f})"
        )
    else:
        logger.info(
            f"Using provided decision threshold: {threshold_to_use:.4f} "
            f"(Youden best={youden_threshold:.4f}, J={youden_j:.4f})"
        )

    # Threshold scores to derive confusion matrix and per-class metrics
    pred_labels = (all_scores >= threshold_to_use).astype(np.int32)

    tp = int(((pred_labels == 1) & (all_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (all_labels == 0)).sum())
    fp = int(((pred_labels == 1) & (all_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (all_labels == 1)).sum())

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, labels=[0, 1], zero_division=0
    )

    metrics = FrameLevelMetrics(
        frame_auc=float(frame_auc),
        num_frames=len(all_labels),
        num_videos=len(valid_videos),
        decision_threshold=float(threshold_to_use),
        youden_j=float(youden_j),
        youden_threshold=float(youden_threshold),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        anomaly_precision=float(prec[1]),
        anomaly_recall=float(rec[1]),
        anomaly_f1=float(f1[1]),
        normal_precision=float(prec[0]),
        normal_recall=float(rec[0]),
        normal_f1=float(f1[0]),
    )

    curves = FrameLevelCurves(
        fpr=roc_fpr,
        tpr=roc_tpr,
        roc_thresholds=roc_thresholds,
        precision=pr_precision,
        recall=pr_recall,
        pr_thresholds=pr_thresholds,
    )

    return metrics, curves, video_results, all_scores, all_labels


def save_frame_level_results(
    results_dir: Path,
    run_name: str,
    checkpoint_path: Path,
    timestamp: str,
    metrics: FrameLevelMetrics,
    curves: FrameLevelCurves,
    scores: np.ndarray,
    labels: np.ndarray,
    video_results: list,
):
    """Persist frame-level evaluation outputs."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Text summary
    results_file = results_dir / "metrics.txt"
    with open(results_file, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nFrame-Level AUC: {metrics.frame_auc:.4f}\n")
        f.write(f"Decision Threshold: {metrics.decision_threshold:.3f}\n")
        f.write(
            f"Youden J: {metrics.youden_j:.4f} @ Threshold: {metrics.youden_threshold:.3f}\n"
        )
        f.write(f"Total Frames: {metrics.num_frames:,}\n")
        f.write(f"Total Videos: {metrics.num_videos}\n")
        f.write(
            f"Counts: TP={metrics.tp} FN={metrics.fn} | TN={metrics.tn} FP={metrics.fp}\n"
        )
        f.write(
            f"Anomaly - P: {metrics.anomaly_precision:.4f} "
            f"R: {metrics.anomaly_recall:.4f} F1: {metrics.anomaly_f1:.4f}\n"
        )
        f.write(
            f"Normal  - P: {metrics.normal_precision:.4f} "
            f"R: {metrics.normal_recall:.4f} F1: {metrics.normal_f1:.4f}\n"
        )
        f.write(f"FPR: {metrics.fpr:.4f} | FNR: {metrics.fnr:.4f}\n")

    # Raw arrays for plotting/analysis
    confusion = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    raw_npz = results_dir / "raw_data.npz"
    np.savez(
        raw_npz,
        fpr=curves.fpr,
        tpr=curves.tpr,
        roc_thresholds=curves.roc_thresholds,
        precision=curves.precision,
        recall=curves.recall,
        pr_thresholds=curves.pr_thresholds,
        frame_auc=metrics.frame_auc,
        scores=scores,
        labels=labels,
        confusion=confusion,
        decision_threshold=metrics.decision_threshold,
        youden_threshold=metrics.youden_threshold,
        youden_j=metrics.youden_j,
        run_name=run_name,
        timestamp=timestamp,
    )

    # Video-level artifacts
    with open(results_dir / "video_results.pkl", "wb") as f:
        pickle.dump(video_results, f)
