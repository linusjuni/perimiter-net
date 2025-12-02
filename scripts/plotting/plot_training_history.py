import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"training_history.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_out_dir(run_dir: Path, run_name: str) -> Path:
    """
    Save plots under repo_root/plots/<run_name>/...
    """
    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / "plots" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_loss(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss", ax=ax)
    if "val_loss" in df.columns:
        sns.lineplot(data=df, x="epoch", y="val_loss", label="Val Loss", ax=ax)
    ax.set_title("Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)


def plot_accuracy(df: pd.DataFrame, out_dir: Path):
    if "train_acc" not in df.columns and "val_acc" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if "train_acc" in df.columns:
        sns.lineplot(data=df, x="epoch", y="train_acc", label="Train Acc", ax=ax)
    if "val_acc" in df.columns:
        sns.lineplot(data=df, x="epoch", y="val_acc", label="Val Acc", ax=ax)
    ax.set_title("Accuracy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)


def plot_auc(df: pd.DataFrame, out_dir: Path):
    if "val_auc" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="epoch", y="val_auc", label="Val AUC", ax=ax)
    ax.set_title("Validation AUC vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_auc.png", dpi=150)
    plt.close(fig)


def plot_learning_rate(df: pd.DataFrame, out_dir: Path):
    if "learning_rate" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x="epoch", y="learning_rate", label="Learning Rate", ax=ax)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.plotting.plot_training_history <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    csv_path = run_dir / "training_history.csv"

    try:
        df = load_history(csv_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_name = run_dir.name
    out_dir = ensure_out_dir(run_dir, run_name)
    logger.info(f"Loaded history from {csv_path}")
    logger.info(f"Saving plots to: {out_dir}")

    plot_loss(df, out_dir)
    plot_accuracy(df, out_dir)
    plot_auc(df, out_dir)
    plot_learning_rate(df, out_dir)

    logger.info("Finished plotting training history.")


if __name__ == "__main__":
    main()
