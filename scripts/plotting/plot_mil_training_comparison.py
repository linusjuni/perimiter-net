import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


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

    for i, (run_name, df) in enumerate(histories):
        if "epoch" not in df.columns:
            continue
        
        color = colors[i % len(colors)]
        epochs = df["epoch"].values
        
        # Plot train loss (dashed)
        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"].values, 
                    linestyle="--", color=color, linewidth=2,
                    label=f"{run_name} (train)")
        
        # Plot val loss (solid)
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"].values, 
                    linestyle="-", color=color, linewidth=2,
                    label=f"{run_name} (val)")

    ax.set_title("Training vs Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(False)
    
    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / f"{tag}_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare training histories from multiple run directories."
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path, 
        help="Paths to run directories containing training_history.csv"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag for output folder/file name."
    )
    parser.add_argument(
        "--title", type=str, default="Training vs Validation Loss",
        help="Plot title."
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
