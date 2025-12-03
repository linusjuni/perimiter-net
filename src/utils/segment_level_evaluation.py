import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_curve
from tqdm import tqdm

from src.utils.evaluation_utils import compute_auc_safe, compute_youdens_j
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentLevelMetrics:
    """Container for segment-level evaluation results."""

    segment_auc: float
    num_segments: int
    num_videos: int
    decision_threshold: float
    youden_j: float
    youden_threshold: float

    tp: int
    tn: int
    fp: int
    fn: int

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

    def __str__(self) -> str:
        return (
            f"Segment-Level AUC: {self.segment_auc:.4f}\n"
            f"   >> Threshold: {self.decision_threshold:.3f} (Youden J={self.youden_j:.3f} @ {self.youden_threshold:.3f}) | "
            f"Segments: {self.num_segments:,} | Videos: {self.num_videos}\n"
            f"   >> Anomaly: P={self.anomaly_precision:.3f} R={self.anomaly_recall:.3f} F1={self.anomaly_f1:.3f}\n"
            f"   >> Normal:  P={self.normal_precision:.3f} R={self.normal_recall:.3f} F1={self.normal_f1:.3f}\n"
            f"   >> Counts:  TP={self.tp} FN={self.fn} | TN={self.tn} FP={self.fp} | FPR={self.fpr:.4f} FNR={self.fnr:.4f}"
        )


@dataclass
class SegmentLevelCurves:
    """Precomputed curve data for downstream plotting."""

    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    pr_thresholds: np.ndarray


def parse_temporal_annotations(annotation_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse UCF-Crime temporal annotation file into a dict."""
    annotations: Dict[str, List[Tuple[int, int]]] = {}

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

            intervals: List[Tuple[int, int]] = []
            if start1 != -1 and end1 != -1:
                intervals.append((start1, end1))
            if start2 != -1 and end2 != -1:
                intervals.append((start2, end2))

            annotations[video_name] = intervals

    logger.info(f"Parsed annotations for {len(annotations)} videos")
    return annotations


def _interpolate_features(features: np.ndarray, segments: int) -> np.ndarray:
    """Match training-time interpolation: split into `segments` and average each chunk."""
    t, d = features.shape
    if t == segments:
        return features

    chunks = np.array_split(features, segments, axis=0)
    interpolated = np.zeros((segments, d), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        if chunk.shape[0] > 0:
            interpolated[i] = np.mean(chunk, axis=0)
        else:
            interpolated[i] = np.zeros(d, dtype=np.float32)

    return interpolated


def _build_segment_labels(
    num_clips: int,
    annotation_intervals: List[Tuple[int, int]],
    segments: int,
    clip_len: int = 16,
    stride: int = 16,
    frame_start_idx: int = 0,
) -> np.ndarray:
    """Create per-segment labels by checking clip/frame overlap with annotations."""
    clip_labels = np.zeros(num_clips, dtype=np.int32)

    for idx in range(num_clips):
        start = frame_start_idx + idx * stride
        end = start + clip_len - 1

        for ann_start, ann_end in annotation_intervals:
            if start <= ann_end and end >= ann_start:  # overlap
                clip_labels[idx] = 1
                break

    # Downsample/aggregate to segment labels with the same array_split as features
    chunks = np.array_split(clip_labels, segments)
    segment_labels = np.zeros(segments, dtype=np.int32)
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            segment_labels[i] = int(np.max(chunk))
        else:
            segment_labels[i] = 0

    return segment_labels


def evaluate_segment_level(
    model: torch.nn.Module,
    feature_dir: str,
    annotation_path: str,
    device: torch.device,
    segments: int = 32,
    clip_len: int = 16,
    stride: int = 16,
    frame_start_idx: int = 0,
    decision_threshold: Optional[float] = None,
) -> Tuple[SegmentLevelMetrics, SegmentLevelCurves, list, np.ndarray, np.ndarray]:
    """
    Perform segment-level evaluation on UCF-Crime test features.

    Args:
        model: Trained MIL model
        feature_dir: Directory containing .npy feature bags for Test videos
        annotation_path: Path to temporal annotations
        device: Evaluation device
        segments: Number of segments to interpolate to (matches training)
        clip_len: Frames per clip used during feature extraction
        stride: Stride between clips during feature extraction
        frame_start_idx: First frame index used during feature extraction (e.g., 0 or 1)
        decision_threshold: Optional fixed threshold; defaults to Youden's J

    Returns:
        SegmentLevelMetrics, SegmentLevelCurves, video_results list, all_scores, all_labels
    """
    logger.info("Starting segment-level evaluation...")

    annotations = parse_temporal_annotations(Path(annotation_path))

    feature_paths = sorted(Path(feature_dir).glob("*.npy"))
    if not feature_paths:
        raise FileNotFoundError(f"No .npy feature files found in {feature_dir}")

    # Storage
    all_scores: List[float] = []
    all_labels: List[int] = []
    video_results = []  # (video_id, video_auc, seg_scores, seg_labels, intervals)

    model.eval()
    with torch.no_grad():
        for feat_path in tqdm(feature_paths, desc="Processing feature bags"):
            video_id = feat_path.stem
            intervals = annotations.get(video_id, [])

            try:
                features = np.load(feat_path)
            except Exception as e:
                logger.error(f"Failed to load {feat_path.name}: {e}")
                continue

            if features.ndim != 2 or features.shape[0] == 0:
                logger.warning(f"Skipping invalid feature file: {feat_path.name}")
                continue

            num_clips, feat_dim = features.shape

            # Clip-level frame ranges
            clip_starts = frame_start_idx + np.arange(num_clips) * stride
            clip_ends = clip_starts + clip_len - 1

            # Labels
            segment_labels = _build_segment_labels(
                num_clips=num_clips,
                annotation_intervals=intervals,
                segments=segments,
                clip_len=clip_len,
                stride=stride,
                frame_start_idx=frame_start_idx,
            )

            # Segment frame ranges (min/max over clips in each chunk)
            seg_starts: List[float] = []
            seg_ends: List[float] = []
            seg_centers: List[float] = []
            clip_indices_chunks = np.array_split(np.arange(num_clips), segments)
            for chunk in clip_indices_chunks:
                if len(chunk) > 0:
                    start = float(np.min(clip_starts[chunk]))
                    end = float(np.max(clip_ends[chunk]))
                else:
                    start = float(frame_start_idx)
                    end = float(frame_start_idx)
                seg_starts.append(start)
                seg_ends.append(end)
                seg_centers.append((start + end) / 2.0)

            # Features -> segments
            seg_features = _interpolate_features(features, segments=segments)
            seg_tensor = (
                torch.from_numpy(seg_features).float().unsqueeze(0).to(device)
            )  # (1, segments, D)

            # Forward
            seg_scores = model(seg_tensor).squeeze(-1).squeeze(0).detach().cpu().numpy()

            # Per-video AUC if both classes present
            if len(np.unique(segment_labels)) > 1:
                video_auc = compute_auc_safe(segment_labels, seg_scores)
            else:
                video_auc = float("nan")

            # Approximate frame positions for segments (even spacing across total frames)
            total_frames = frame_start_idx + (num_clips - 1) * stride + clip_len
            segment_frame_centers = np.linspace(
                frame_start_idx,
                total_frames,
                num=segments,
                endpoint=False,
                dtype=np.float32,
            )

            video_results.append(
                (
                    video_id,
                    video_auc,
                    seg_scores,
                    segment_labels,
                    seg_centers,
                    list(zip(seg_starts, seg_ends)),
                    intervals,
                )
            )

            all_scores.extend(seg_scores.tolist())
            all_labels.extend(segment_labels.tolist())

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels, dtype=np.int32)

    # Global AUC
    segment_auc = compute_auc_safe(all_labels_np, all_scores_np)
    logger.info(f"Segment-Level AUC: {segment_auc:.4f}")

    # Curves + thresholds
    if len(np.unique(all_labels_np)) > 1:
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(all_labels_np, all_scores_np)
        youden_threshold, youden_j = compute_youdens_j(roc_fpr, roc_tpr, roc_thresholds)
    else:
        roc_fpr = np.array([])
        roc_tpr = np.array([])
        roc_thresholds = np.array([])
        youden_threshold = float("nan")
        youden_j = float("nan")

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        all_labels_np, all_scores_np
    )

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

    pred_labels = (all_scores_np >= threshold_to_use).astype(np.int32)

    tp = int(((pred_labels == 1) & (all_labels_np == 1)).sum())
    tn = int(((pred_labels == 0) & (all_labels_np == 0)).sum())
    fp = int(((pred_labels == 1) & (all_labels_np == 0)).sum())
    fn = int(((pred_labels == 0) & (all_labels_np == 1)).sum())

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels_np, pred_labels, labels=[0, 1], zero_division=0
    )

    metrics = SegmentLevelMetrics(
        segment_auc=float(segment_auc),
        num_segments=len(all_labels_np),
        num_videos=len(feature_paths),
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

    curves = SegmentLevelCurves(
        fpr=roc_fpr,
        tpr=roc_tpr,
        roc_thresholds=roc_thresholds,
        precision=pr_precision,
        recall=pr_recall,
        pr_thresholds=pr_thresholds,
    )

    return metrics, curves, video_results, all_scores_np, all_labels_np


def save_segment_level_results(
    results_dir: Path,
    run_name: str,
    checkpoint_path: Path,
    timestamp: str,
    metrics: SegmentLevelMetrics,
    curves: SegmentLevelCurves,
    scores: np.ndarray,
    labels: np.ndarray,
    video_results: list,
):
    """Persist segment-level evaluation outputs (metrics, raw arrays, video results)."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Text summary
    results_file = results_dir / "metrics.txt"
    with open(results_file, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nSegment-Level AUC: {metrics.segment_auc:.4f}\n")
        f.write(f"Decision Threshold: {metrics.decision_threshold:.3f}\n")
        f.write(
            f"Youden J: {metrics.youden_j:.4f} @ Threshold: {metrics.youden_threshold:.3f}\n"
        )
        f.write(f"Total Segments: {metrics.num_segments:,}\n")
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
        segment_auc=metrics.segment_auc,
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
    with open(results_dir / "video_results.json", "w") as f:
        json.dump(
            [
                {
                    "video_id": vid,
                    "video_auc": auc,
                    "intervals": intervals,
                }
                for vid, auc, _, _, _, _, intervals in video_results
            ],
            f,
            indent=2,
        )

    # Full per-video scores/labels/positions (for analysis/plots)
    np.save(results_dir / "video_results.npy", np.array(video_results, dtype=object))

    logger.info(f"Saved segment-level results to {results_dir}")
