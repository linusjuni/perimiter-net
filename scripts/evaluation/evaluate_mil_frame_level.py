"""
Grid search with k-fold cross-validation for MIL video-level models and late fusion.

Runs the same hyperparameter grid on RGB and motion MIL models, reports mean/std AUC
across folds, and optionally evaluates late fusion (weighted average) on each fold.
"""

import argparse
import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root on path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.mil import MILDataLoader
from src.models.mil import MILModel
from src.utils.losses import MILRankingLoss
from src.utils.mil_evaluation import evaluate_mil
from src.utils.training import train_epoch_mil
from src.utils.training_utils import EarlyStopping
from src.utils.evaluation_utils import compute_auc_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoEntry:
    video_id: str
    label: int  # 0 = normal, 1 = anomaly
    rgb_path: Path | None
    motion_path: Path | None


@dataclass
class HyperParams:
    lr: float
    weight_decay: float
    lambda1: float
    lambda2: float
    dropout: float
    input_dim: int = 512


@dataclass
class FoldResult:
    auc: float
    y_true: np.ndarray
    y_scores: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search + cross-validation for MIL models (RGB, motion, fusion)"
    )
    parser.add_argument("--rgb_features", type=str, required=True, help="RGB feature dir")
    parser.add_argument(
        "--motion_features", type=str, required=True, help="Motion feature dir"
    )
    parser.add_argument("--output_dir", type=str, default="results/mil_grid_search")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Evaluate every N epochs (also evaluates at epoch 1 and final)",
    )
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameter grid
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4],
        help="Learning rates to try",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        nargs="+",
        default=[5e-3, 1e-4],
        help="Weight decay values to try",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        nargs="+",
        default=[8e-5, 1e-4],
        help="Sparsity coefficient grid",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        nargs="+",
        default=[8e-5, 1e-4],
        help="Smoothness coefficient grid",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        nargs="+",
        default=[0.5, 0.6],
        help="Dropout probabilities for MIL head",
    )
    parser.add_argument(
        "--fusion_weights",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7],
        help="Alpha weights for fusion: score = alpha*rgb + (1-alpha)*motion",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_feature_dir(feature_dir: Path) -> dict[str, Tuple[Path, int]]:
    index = {}
    for path in feature_dir.glob("*.npy"):
        label = 0 if "Normal" in path.name else 1
        index[path.stem] = (path, label)
    return index


def collect_entries(rgb_dir: Path, motion_dir: Path) -> List[VideoEntry]:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB feature dir not found: {rgb_dir}")
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion feature dir not found: {motion_dir}")

    rgb_index = scan_feature_dir(rgb_dir)
    motion_index = scan_feature_dir(motion_dir)

    common_ids = sorted(set(rgb_index.keys()) & set(motion_index.keys()))
    if not common_ids:
        raise ValueError("No overlapping videos between RGB and motion feature dirs.")

    entries: List[VideoEntry] = []
    for vid in common_ids:
        rgb_path, rgb_label = rgb_index[vid]
        motion_path, motion_label = motion_index[vid]

        if rgb_label != motion_label:
            logger.warning(f"Label mismatch for {vid}, skipping.")
            continue

        entries.append(VideoEntry(vid, rgb_label, rgb_path, motion_path))

    logger.info(
        f"Using {len(entries)} videos (RGB={len(rgb_index)}, Motion={len(motion_index)})"
    )
    return entries


def build_folds(entries: List[VideoEntry], k: int, seed: int):
    labels = np.array([e.label for e in entries])
    indices = np.arange(len(entries))

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(indices, labels))


def split_files(
    entries: List[VideoEntry], idx: Iterable[int], modality: str
) -> Tuple[list[Path], list[Path]]:
    normal, anomaly = [], []
    for i in idx:
        entry = entries[i]
        path = entry.rgb_path if modality == "rgb" else entry.motion_path
        if path is None:
            continue
        (normal if entry.label == 0 else anomaly).append(path)

    normal = sorted(normal, key=lambda p: p.name)
    anomaly = sorted(anomaly, key=lambda p: p.name)
    return normal, anomaly


def train_and_eval_stream(
    train_normal: list[Path],
    train_anomaly: list[Path],
    val_normal: list[Path],
    val_anomaly: list[Path],
    params: HyperParams,
    device: torch.device,
    max_epochs: int,
    eval_interval: int,
    batch_size: int,
    segments: int,
    patience: int,
    feature_dir: Path,
    split_name: str,
) -> FoldResult:
    train_loader = MILDataLoader(
        feature_dir=feature_dir,
        segments=segments,
        shuffle=True,
        split="train",
        val_split=0.0,
        normal_files=train_normal,
        anomaly_files=train_anomaly,
    )
    val_loader = MILDataLoader(
        feature_dir=feature_dir,
        segments=segments,
        shuffle=False,
        split="val",
        val_split=0.0,
        normal_files=val_normal,
        anomaly_files=val_anomaly,
    )

    model = MILModel(input_dim=params.input_dim, dropout=params.dropout).to(device)
    criterion = MILRankingLoss(lambda_1=params.lambda1, lambda_2=params.lambda2)
    optimizer = AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=patience, mode="max")

    best_auc = -float("inf")
    best_scores: tuple[np.ndarray, np.ndarray] | None = None

    # Adjust batch size if a fold is small
    effective_batch = min(
        batch_size, len(train_loader.normal_videos), len(train_loader.anomaly_videos)
    )
    effective_batch = max(effective_batch, 1)

    for epoch in range(1, max_epochs + 1):
        train_epoch_mil(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            batch_size=effective_batch,
        )

        if epoch == 1 or epoch % eval_interval == 0 or epoch == max_epochs:
            val_metrics, y_true, y_scores = evaluate_mil(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                split=f"{split_name}_val",
                return_scores=True,
            )

            if val_metrics.auc > best_auc:
                best_auc = val_metrics.auc
                best_scores = (y_true, y_scores)

            stopper(val_metrics.auc)
            if stopper.early_stop:
                logger.info(
                    f"[{split_name}] Early stopping at epoch {epoch} (AUC={val_metrics.auc:.4f})"
                )
                break

        scheduler.step()

    if best_scores is None:
        # Ensure we always return something
        val_metrics, y_true, y_scores = evaluate_mil(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split=f"{split_name}_val",
            return_scores=True,
        )
        best_auc = val_metrics.auc
        best_scores = (y_true, y_scores)

    return FoldResult(auc=best_auc, y_true=best_scores[0], y_scores=best_scores[1])


def fusion_auc(
    y_true: np.ndarray,
    rgb_scores: np.ndarray,
    motion_scores: np.ndarray,
    weights: Iterable[float],
) -> list[tuple[float, float]]:
    results = []
    for alpha in weights:
        fused = alpha * rgb_scores + (1 - alpha) * motion_scores
        auc = compute_auc_safe(y_true, fused)
        results.append((alpha, auc))
    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    rgb_dir = Path(args.rgb_features)
    motion_dir = Path(args.motion_features)
    entries = collect_entries(rgb_dir, motion_dir)
    folds = build_folds(entries, args.folds, args.seed)

    grid = list(
        itertools.product(
            args.lr, args.weight_decay, args.lambda1, args.lambda2, args.dropout
        )
    )
    logger.info(f"Total hyperparameter combinations: {len(grid)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"mil_grid_search_{timestamp}.csv"

    # Prepare CSV writer
    import csv

    fieldnames = [
        "stream",
        "fusion_weight",
        "lr",
        "weight_decay",
        "lambda1",
        "lambda2",
        "dropout",
        "mean_auc",
        "std_auc",
        "fold_aucs",
    ]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (lr, wd, l1, l2, dr) in enumerate(grid, start=1):
            params = HyperParams(lr=lr, weight_decay=wd, lambda1=l1, lambda2=l2, dropout=dr)
            logger.info(
                f"[{idx}/{len(grid)}] lr={lr} wd={wd} lambda1={l1} lambda2={l2} dropout={dr}"
            )

            rgb_fold_aucs = []
            motion_fold_aucs = []
            fusion_per_alpha = {alpha: [] for alpha in args.fusion_weights}

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_normal_rgb, train_anom_rgb = split_files(entries, train_idx, "rgb")
                val_normal_rgb, val_anom_rgb = split_files(entries, val_idx, "rgb")

                train_normal_motion, train_anom_motion = split_files(
                    entries, train_idx, "motion"
                )
                val_normal_motion, val_anom_motion = split_files(
                    entries, val_idx, "motion"
                )

                rgb_result = train_and_eval_stream(
                    train_normal_rgb,
                    train_anom_rgb,
                    val_normal_rgb,
                    val_anom_rgb,
                    params,
                    device,
                    args.epochs,
                    args.eval_interval,
                    args.batch_size,
                    args.segments,
                    args.patience,
                    rgb_dir,
                    split_name=f"rgb_fold{fold_idx+1}",
                )
                rgb_fold_aucs.append(rgb_result.auc)

                motion_result = train_and_eval_stream(
                    train_normal_motion,
                    train_anom_motion,
                    val_normal_motion,
                    val_anom_motion,
                    params,
                    device,
                    args.epochs,
                    args.eval_interval,
                    args.batch_size,
                    args.segments,
                    args.patience,
                    motion_dir,
                    split_name=f"motion_fold{fold_idx+1}",
                )
                motion_fold_aucs.append(motion_result.auc)

                # Late fusion on the shared validation set
                if len(rgb_result.y_true) != len(motion_result.y_true):
                    logger.warning(
                        f"Fold {fold_idx+1}: mismatched val sizes (rgb={len(rgb_result.y_true)}, motion={len(motion_result.y_true)}), skipping fusion."
                    )
                else:
                    for alpha, auc in fusion_auc(
                        rgb_result.y_true,
                        rgb_result.y_scores,
                        motion_result.y_scores,
                        args.fusion_weights,
                    ):
                        fusion_per_alpha[alpha].append(auc)

            # Save per-stream summaries
            writer.writerow(
                {
                    "stream": "rgb",
                    "fusion_weight": "",
                    "lr": lr,
                    "weight_decay": wd,
                    "lambda1": l1,
                    "lambda2": l2,
                    "dropout": dr,
                    "mean_auc": float(np.mean(rgb_fold_aucs)),
                    "std_auc": float(np.std(rgb_fold_aucs)),
                    "fold_aucs": ";".join(f"{x:.4f}" for x in rgb_fold_aucs),
                }
            )
            writer.writerow(
                {
                    "stream": "motion",
                    "fusion_weight": "",
                    "lr": lr,
                    "weight_decay": wd,
                    "lambda1": l1,
                    "lambda2": l2,
                    "dropout": dr,
                    "mean_auc": float(np.mean(motion_fold_aucs)),
                    "std_auc": float(np.std(motion_fold_aucs)),
                    "fold_aucs": ";".join(f"{x:.4f}" for x in motion_fold_aucs),
                }
            )

            for alpha, aucs in fusion_per_alpha.items():
                if not aucs:
                    continue
                writer.writerow(
                    {
                        "stream": "fusion",
                        "fusion_weight": alpha,
                        "lr": lr,
                        "weight_decay": wd,
                        "lambda1": l1,
                        "lambda2": l2,
                        "dropout": dr,
                        "mean_auc": float(np.mean(aucs)),
                        "std_auc": float(np.std(aucs)),
                        "fold_aucs": ";".join(f"{x:.4f}" for x in aucs),
                    }
                )

    logger.info(f"Grid search complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
