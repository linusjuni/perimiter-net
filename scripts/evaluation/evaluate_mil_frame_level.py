import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel
from src.utils.evaluation_utils import compute_auc_safe
from src.utils.training_utils import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_checkpoint(path: Path) -> Optional[Path]:
    """Resolve checkpoint path from file or directory."""
    if path.is_file():
        return path

    if path.is_dir():
        best = path / "best_model.pth"
        if best.exists():
            return best

        epoch_ckpts = sorted(path.glob("checkpoint_epoch_*.pth"))
        if epoch_ckpts:
            return epoch_ckpts[-1]

    return None


def find_available_checkpoints(base_dir: Path) -> List[Tuple[str, Path]]:
    """Find available MIL checkpoints in base directory."""
    checkpoints: List[Tuple[str, Path]] = []
    if not base_dir.exists():
        return checkpoints

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        ckpt = resolve_checkpoint(run_dir)
        if ckpt is not None:
            checkpoints.append((run_dir.name, ckpt))

    return checkpoints


def select_checkpoint(checkpoints: List[Tuple[str, Path]]) -> Optional[Path]:
    """Interactive selection of a checkpoint."""
    if not checkpoints:
        logger.error("No checkpoints found to select from.")
        return None

    print("\n" + "=" * 60)
    print("Available MIL Checkpoints:")
    print("=" * 60)
    for idx, (run_name, ckpt_path) in enumerate(checkpoints, start=1):
        print(f"[{idx:02d}] {run_name} -> {ckpt_path.name}")
    print("=" * 60)

    try:
        choice = input(f"Select checkpoint (1-{len(checkpoints)}): ").strip()
        sel_idx = int(choice) - 1
        if sel_idx < 0 or sel_idx >= len(checkpoints):
            raise ValueError
    except (ValueError, KeyboardInterrupt):
        logger.error("Invalid selection; aborting.")
        return None

    return checkpoints[sel_idx][1]


def load_temporal_annotations(annotation_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse UCF-Crime temporal annotations."""
    annotations: Dict[str, List[Tuple[int, int]]] = {}

    if not annotation_path.exists():
        logger.warning(f"Annotation file not found: {annotation_path}")
        return annotations

    with open(annotation_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                logger.warning(f"Skipping malformed line: {line}")
                continue

            video_id = parts[0].replace(".mp4", "")

            try:
                start1, end1, start2, end2 = map(int, parts[2:6])
            except ValueError:
                logger.warning(f"Could not parse frame indices for {video_id}: {line}")
                continue

            intervals: List[Tuple[int, int]] = []
            if start1 >= 0 and end1 >= 0 and end1 >= start1:
                intervals.append((start1, end1))
            if start2 >= 0 and end2 >= 0 and end2 >= start2:
                intervals.append((start2, end2))

            annotations[video_id] = intervals

    logger.info(f"Loaded annotations for {len(annotations)} videos")
    return annotations


def build_ground_truth_mask(length: int, intervals: List[Tuple[int, int]]) -> np.ndarray:
    """Create a mask for intervals, clamped to length."""
    mask = np.zeros(length, dtype=np.int32)

    for start, end in intervals:
        if end < 0 or start >= length:
            continue
        s = max(0, start)
        e = min(length - 1, end)
        mask[s : e + 1] = 1

    return mask


def run_inference(
    model: torch.nn.Module, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Run MIL model on feature matrix."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")

    feats_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(feats_tensor)  # (1, T, 1)

    scores = scores.squeeze(0).squeeze(-1).detach().cpu().numpy()
    return scores


def main():
    # Configuration
    feature_dir = Path("/work3/s225224/ucf-crime/features/rgb/Test")
    base_checkpoint_dir = Path("/work3/s225224/ucf-crime/checkpoints/mil")
    annotation_file = Path(
        "/work3/s225224/ucf-crime/data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    )
    results_base_dir = Path("/work3/s225224/ucf-crime/experiments/mil_frame_level")
    stride = 16  # Frames represented by each clip score
    sigma = 16  # Gaussian smoothing sigma for the expanded frame scores
    input_dim = 512
    # End configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    logger.info("MIL Frame-Level Evaluation")
    logger.info("=" * 80)
    logger.info(f"Feature dir     : {feature_dir}")
    logger.info(f"Checkpoint root : {base_checkpoint_dir}")
    logger.info(f"Annotation file : {annotation_file}")
    logger.info(f"Results dir     : {results_base_dir}")
    logger.info(f"Stride / Sigma  : {stride} / {sigma}")
    logger.info(f"Device          : {device}")
    logger.info("=" * 80)

    if not feature_dir.exists():
        logger.error(f"Feature directory not found: {feature_dir}")
        return

    checkpoints = find_available_checkpoints(base_checkpoint_dir)
    resolved_checkpoint = select_checkpoint(checkpoints)
    if resolved_checkpoint is None:
        return
    logger.info(f"Using checkpoint file: {resolved_checkpoint}")

    feature_paths = sorted(feature_dir.glob("*.npy"))
    if not feature_paths:
        logger.error(f"No .npy feature files found in {feature_dir}")
        return

    # Load annotations
    annotations = load_temporal_annotations(annotation_file)

    # Load model and checkpoint
    model = MILModel(input_dim=input_dim).to(device)
    load_checkpoint(resolved_checkpoint, model, device=device)

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_video: List[Dict[str, object]] = []

    for feature_path in feature_paths:
        video_id = feature_path.stem
        intervals = annotations.get(video_id, [])
        if video_id not in annotations:
            logger.warning(f"No annotation entry for {video_id}; treating as Normal")

        try:
            features = np.load(feature_path)
        except Exception as exc:
            logger.error(f"Failed to load {feature_path.name}: {exc}")
            continue

        if features.ndim != 2:
            logger.error(
                f"{video_id}: expected 2D features, got array with shape {features.shape}"
            )
            continue

        if features.shape[1] != input_dim:
            logger.warning(
                f"{video_id}: expected feature dim {input_dim}, got {features.shape[1]}"
            )

        try:
            clip_scores = run_inference(model, features, device)
        except Exception as exc:
            logger.error(f"Inference failed for {video_id}: {exc}")
            continue

        # Map clip scores to frame timeline
        frame_scores = np.repeat(clip_scores, stride)

        # Smooth the signal
        smoothed_scores = gaussian_filter1d(frame_scores, sigma=sigma)

        # Ground truth mask
        gt_mask = build_ground_truth_mask(len(smoothed_scores), intervals)

        video_auc = None
        if len(np.unique(gt_mask)) > 1:
            video_auc = compute_auc_safe(gt_mask, smoothed_scores)

        # Accumulate results
        all_scores.append(smoothed_scores.astype(np.float32))
        all_labels.append(gt_mask.astype(np.int32))
        per_video.append(
            {
                "video_id": video_id,
                "num_clips": int(len(clip_scores)),
                "num_frames": int(len(smoothed_scores)),
                "video_auc": float(video_auc) if video_auc is not None else None,
            }
        )

        if video_auc is not None:
            logger.info(
                f"{video_id}: clips={len(clip_scores):3d} frames={len(smoothed_scores):5d} AUC={video_auc:.4f}"
            )
        else:
            logger.info(
                f"{video_id}: clips={len(clip_scores):3d} frames={len(smoothed_scores):5d} (single-class GT)"
            )

    if not all_scores:
        logger.error("No videos were processed successfully.")
        return

    y_scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    frame_auc = compute_auc_safe(y_true, y_scores)

    logger.info("=" * 80)
    logger.info(f"Frame-Level AUC: {frame_auc:.4f}")
    logger.info(f"Total frames aggregated: {len(y_true):,}")
    logger.info("=" * 80)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = resolved_checkpoint.parent.name
    results_dir = results_base_dir / f"{run_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Checkpoint: {resolved_checkpoint}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Frame-Level AUC: {frame_auc:.6f}\n")
        f.write(f"Total Frames: {len(y_true)}\n")
        f.write(f"Total Videos: {len(per_video)}\n")

    np.savez(
        results_dir / "raw_data.npz",
        scores=y_scores,
        labels=y_true,
        frame_auc=frame_auc,
    )

    with open(results_dir / "per_video.json", "w") as f:
        json.dump(per_video, f, indent=2)

    logger.info(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
