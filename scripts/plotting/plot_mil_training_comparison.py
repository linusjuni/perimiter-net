import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")


def clean_run_name(run_name: str) -> str:
    """Extract a clean label from run name by removing timestamp suffix."""
    import re

    # Remove timestamp pattern like _20251203_234217
    cleaned = re.sub(r"_\d{8}_\d{6}$", "", run_name)
    return cleaned


def friendly_label(run_name: str) -> str:
    """Convert run name to friendly label like 'MIL RGB' or 'MIL Motion'."""
    cleaned = clean_run_name(run_name).lower()
    if "rgb" in cleaned:
        return "MIL RGB"
    if "motion" in cleaned:
        return "MIL Motion"
    # Fallback: capitalize words
    return " ".join(
        word.upper() if word in ("mil", "rgb") else word.capitalize()
        for word in cleaned.replace("_", " ").split()
    )


def load_history(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "training_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"training_history.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["run_name"] = run_dir.name
    return df


def ensure_out_dir(tag: str) -> Path:
    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / "plots" / "mil" / "comparisons" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_comparison(histories: List[Tuple[str, pd.DataFrame]], out_dir: Path, tag: str):
    if not histories:
        logger.error("No histories to plot.")
        return

    # Use a clean color palette
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))

    best_epochs = []  # Store (epoch, color, label) for vertical lines

    for i, (run_name, df) in enumerate(histories):
        if "epoch" not in df.columns:
            continue

        color = colors[i % len(colors)]
        epochs = df["epoch"].values
        label = friendly_label(run_name)

        # Plot train loss (dashed)
        if "train_loss" in df.columns:
            ax.plot(
                epochs,
                df["train_loss"].values,
                linestyle="--",
                color=color,
                linewidth=2,
                label=f"{label} (Train)",
            )

        # Plot val loss (solid)
        if "val_loss" in df.columns:
            val_loss = df["val_loss"].values
            ax.plot(
                epochs,
                val_loss,
                linestyle="-",
                color=color,
                linewidth=2,
                label=f"{label} (Val)",
            )

            # Find best epoch (lowest val loss)
            best_idx = val_loss.argmin()
            best_epoch = epochs[best_idx]
            best_epochs.append((best_epoch, color, label))

    # Add vertical lines for best epochs
    y_min, y_max = ax.get_ylim()
    for best_epoch, color, label in best_epochs:
        ax.axvline(best_epoch, color=color, linestyle=":", linewidth=2, alpha=0.8)
        ax.text(
            best_epoch + 1,
            y_max * 0.95,
            f"Best {label}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=12,
            color=color,
        )

    ax.set_title("Training vs Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)

    fig.tight_layout()
    out_path = out_dir / f"{tag}_comparison.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare training histories from multiple run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Paths to run directories containing training_history.csv",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for output folder/file name.",
    )
    parser.add_argument(
        "--title", type=str, default="Training vs Validation Loss", help="Plot title."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    histories: List[Tuple[str, pd.DataFrame]] = []
    for run_dir in args.run_dirs:
        try:
            df = load_history(run_dir)
            histories.append((run_dir.name, df))
        except FileNotFoundError:
            logger.warning(f"Skipped: {run_dir}")

    if not histories:
        logger.error("No valid training histories to plot.")
        sys.exit(1)

    tag = args.tag or "__".join([name for name, _ in histories])
    out_dir = ensure_out_dir(tag)
    plot_comparison(histories, out_dir, tag)


if __name__ == "__main__":
    main()
