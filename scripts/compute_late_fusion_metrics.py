import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mil import MILModel
from src.utils.evaluation_utils import compute_youdens_j


def load_temporal_annotations(annotation_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse UCF-Crime temporal annotations.

    Format per line: VideoName Class Start1 End1 Start2 End2
    Where -1 indicates no event for that slot.
    """
    annotations: Dict[str, List[Tuple[int, int]]] = {}

    if not annotation_path.exists():
        print(f"[WARN] Annotation file not found: {annotation_path}")
        return annotations

    with open(annotation_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                print(f"[WARN] Skipping malformed line: {line}")
                continue

            video_id = parts[0].replace(".mp4", "")

            try:
                start1, end1, start2, end2 = map(int, parts[2:6])
            except ValueError:
                print(f"[WARN] Could not parse frame indices for {video_id}: {line}")
                continue

            intervals: List[Tuple[int, int]] = []
            if start1 >= 0 and end1 >= 0 and end1 >= start1:
                intervals.append((start1, end1))
            if start2 >= 0 and end2 >= 0 and end2 >= start2:
                intervals.append((start2, end2))

            annotations[video_id] = intervals

    return annotations


def build_ground_truth_mask(length: int, intervals: List[Tuple[int, int]]) -> np.ndarray:
    """Create a 0/1 mask for the given intervals, clamped to the provided length."""
    mask = np.zeros(length, dtype=np.int32)

    for start, end in intervals:
        if end < 0 or start >= length:
            continue
        s = max(0, start)
        e = min(length - 1, end)
        mask[s : e + 1] = 1

    return mask


def safe_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_scores))
    except ValueError:
        return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return float("nan")


def load_model(ckpt_path: Path, input_dim: int, device: torch.device) -> torch.nn.Module:
    model = MILModel(input_dim=input_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Run MIL model on 2D feature array."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    feats_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(feats_tensor)  # (1, T, 1)
    return scores.squeeze().cpu().numpy()


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray):
    roc_auc = safe_roc_auc(y_true, y_scores)
    pr_auc = safe_average_precision(y_true, y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if thresholds.size > 0:
        f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = int(np.argmax(f1_values))
        best_threshold = float(thresholds[best_idx])
        best_f1_scan = float(f1_values[best_idx])
    else:
        best_threshold = 0.5
        best_f1_scan = float("nan")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    youden_threshold, youden_j = compute_youdens_j(fpr, tpr, roc_thresholds)
    if np.isnan(youden_threshold):
        youden_threshold = 0.5
        youden_j = float("nan")

    y_pred = (y_scores >= youden_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1,
        "best_threshold": youden_threshold,
        "youdens_j": youden_j,
        "best_f1_scan": best_f1_scan,
        "confusion_matrix": cm.tolist(),
        "positive_rate": float(y_true.mean()) if len(y_true) > 0 else float("nan"),
        "f1_threshold_from_pr": best_threshold,
    }


def count_frame_lengths(frames_root: Path) -> Dict[str, int]:
    """
    Count number of extracted frames per video under the provided root.

    Expects structure Test/<class>/*.png with filenames like {VideoId}_x264_{FrameNum}.png.
    """
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")

    counts: Dict[str, int] = {}
    for png in frames_root.rglob("*.png"):
        vid = png.name.rsplit("_", 1)[0]
        counts[vid] = counts.get(vid, 0) + 1
    return counts


def parse_args():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Late fusion (RGB + Motion) for MIL with frame-level metrics."
    )
    parser.add_argument(
        "--rgb-features",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/features/rgb/Test"),
        help="Directory with RGB .npy feature files.",
    )
    parser.add_argument(
        "--motion-features",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/features/motion/Test"),
        help="Directory with Motion .npy feature files.",
    )
    parser.add_argument(
        "--rgb-ckpt",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251204_130129/best_model.pth"),
        help="Checkpoint for RGB MIL model.",
    )
    parser.add_argument(
        "--motion-ckpt",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251204_130253/best_model.pth"),
        help="Checkpoint for Motion MIL model.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
        help="Temporal annotations file (UCF-Crime format).",
    )
    parser.add_argument("--weight-rgb", type=float, default=0.95, help="Late-fusion weight for RGB scores.")
    parser.add_argument("--weight-motion", type=float, default=0.05, help="Late-fusion weight for Motion scores.")
    parser.add_argument("--stride", type=int, default=16, help="Frames represented by each clip score.")
    parser.add_argument("--sigma", type=float, default=16.0, help="Gaussian smoothing sigma on frame scores.")
    parser.add_argument("--input-dim", type=int, default=512, help="Feature dimension for MIL model.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "results" / "late_fusion",
        help="Output directory (predictions + metrics will be placed in a timestamped subfolder).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to include in the output folder name.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("/work3/s225224/ucf-crime/data/Test"),
        help="Root of extracted test frames to align to true frame counts (improves positive_rate).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fusion_tag = args.tag or f"rgb{int(args.weight_rgb * 100):03d}_motion{int(args.weight_motion * 100):03d}"
    run_dir = args.out_dir / f"mil_late_fusion_{fusion_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIL Late Fusion (RGB + Motion) with Frame-level Metrics")
    print("=" * 80)
    print(f"RGB checkpoint    : {args.rgb_ckpt}")
    print(f"Motion checkpoint : {args.motion_ckpt}")
    print(f"Features (RGB)    : {args.rgb_features}")
    print(f"Features (Motion) : {args.motion_features}")
    print(f"Weights RGB/Mot   : {args.weight_rgb:.2f} / {args.weight_motion:.2f}")
    print(f"Annotation file   : {args.annotation_file}")
    print(f"Stride / Sigma    : {args.stride} / {args.sigma}")
    print(f"Device            : {device}")
    print(f"Output dir        : {run_dir}")
    print("=" * 80)

    # Load models
    model_rgb = load_model(args.rgb_ckpt, args.input_dim, device)
    model_motion = load_model(args.motion_ckpt, args.input_dim, device)

    # Locate feature files
    rgb_files = {f.stem: f for f in args.rgb_features.glob("*.npy")}
    motion_files = {f.stem: f for f in args.motion_features.glob("*.npy")}
    common_videos = sorted(list(set(rgb_files.keys()) & set(motion_files.keys())))
    print(f"Found {len(common_videos)} videos common to both streams.")

    annotations = load_temporal_annotations(args.annotation_file)
    if not annotations:
        print("[WARN] No annotations loaded; metrics may be degenerate if GT missing.")

    print(f"Counting frames under: {args.frames_root} (this can take a moment)...")
    frame_counts = count_frame_lengths(args.frames_root)
    print(f"Found frame counts for {len(frame_counts)} videos.")

    predictions: Dict[str, np.ndarray] = {}
    per_video: List[Dict[str, object]] = []
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    skipped_missing_frames = 0
    skipped_missing_ann = 0

    print("Running inference, fusion, and frame-level scoring...")
    for vid_name in tqdm(common_videos):
        try:
            feat_rgb = np.load(rgb_files[vid_name])
            feat_motion = np.load(motion_files[vid_name])

            min_len = min(feat_rgb.shape[0], feat_motion.shape[0])
            if min_len == 0:
                continue

            feat_rgb = feat_rgb[:min_len]
            feat_motion = feat_motion[:min_len]

            scores_rgb = run_inference(model_rgb, feat_rgb, device)
            scores_motion = run_inference(model_motion, feat_motion, device)

            fused_scores = args.weight_rgb * scores_rgb + args.weight_motion * scores_motion
            predictions[vid_name] = fused_scores

            # Frame-level projection and smoothing
            if vid_name not in frame_counts:
                skipped_missing_frames += 1
                continue

            frame_len = frame_counts[vid_name]
            clip_positions = np.linspace(0, frame_len - 1, num=len(fused_scores))
            frame_indices = np.arange(frame_len)
            interp_scores = np.interp(frame_indices, clip_positions, fused_scores)
            smoothed_scores = gaussian_filter1d(interp_scores, sigma=args.sigma)

            intervals = annotations.get(vid_name)
            if intervals is None:
                skipped_missing_ann += 1
                continue

            gt_mask = build_ground_truth_mask(frame_len, intervals)

            all_scores.append(smoothed_scores.astype(np.float32))
            all_labels.append(gt_mask.astype(np.int32))

            video_auc = None
            if len(np.unique(gt_mask)) > 1:
                try:
                    video_auc = roc_auc_score(gt_mask, smoothed_scores)
                except ValueError:
                    video_auc = None

            per_video.append(
                {
                    "video_id": vid_name,
                    "num_clips": int(len(fused_scores)),
                    "num_frames": int(len(smoothed_scores)),
                    "video_auc": float(video_auc) if video_auc is not None else None,
                }
            )
        except Exception as exc:
            print(f"[ERROR] {vid_name}: {exc}")

    if not all_scores:
        print("No videos processed successfully; exiting.")
        return

    y_scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_scores)

    print("=" * 80)
    print("Frame-level Metrics (fused)")
    print(f"ROC AUC      : {metrics['roc_auc']:.4f}")
    print(f"PR  AUC      : {metrics['pr_auc']:.4f}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"F1 (thr opt) : {metrics['f1']:.4f}")
    print(f"Thr (Youden) : {metrics['best_threshold']:.4f} | J={metrics['youdens_j']:.4f}")
    print(f"Best F1 scan : {metrics['best_f1_scan']:.4f} (PR sweep)")
    print(f"Pos rate     : {metrics['positive_rate']:.6f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]")
    print(metrics["confusion_matrix"])
    print("=" * 80)
    print(f"Videos processed: {len(per_video)} | Skipped (no frames): {skipped_missing_frames} | Skipped (no ann): {skipped_missing_ann}")
    print(f"GT positives total: {int(y_true.sum())} / {len(y_true)} ({metrics['positive_rate']:.6f})")

    # Save artifacts
    predictions_path = run_dir / "predictions.pkl"
    with open(predictions_path, "wb") as f:
        pickle.dump(predictions, f)

    metrics_payload = {
        "timestamp": timestamp,
        "rgb_checkpoint": str(args.rgb_ckpt),
        "motion_checkpoint": str(args.motion_ckpt),
        "weight_rgb": args.weight_rgb,
        "weight_motion": args.weight_motion,
        "stride": args.stride,
        "sigma": args.sigma,
        "annotation_file": str(args.annotation_file),
        "metrics": metrics,
        "per_video": per_video,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    np.savez_compressed(run_dir / "frame_scores_labels.npz", scores=y_scores, labels=y_true)

    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved metrics to    : {run_dir / 'metrics.json'}")
    print(f"Saved raw arrays to : {run_dir / 'frame_scores_labels.npz'}")


if __name__ == "__main__":
    main()
