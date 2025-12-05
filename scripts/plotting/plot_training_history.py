import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global seaborn style
sns.set_style("whitegrid")
sns.set_palette("muted")


def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"training_history.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_out_dir(run_dir: Path, run_name: str) -> Path:
    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / "plots" / run_name
    return out_dir


def plot_loss(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns:
        return
    data = df[["epoch", "train_loss"]].copy() if "train_loss" in df.columns else None
    val_data = df[["epoch", "val_loss"]].copy() if "val_loss" in df.columns else None
    if data is not None:
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if val_data is not None:
        val_data = val_data.apply(pd.to_numeric, errors="coerce").dropna()
    if (data is None or data.empty) and (val_data is None or val_data.empty):
        logger.warning("Skipping loss plot: no loss data found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if data is not None and not data.empty:
        sns.lineplot(data=data, x="epoch", y="train_loss", label="Train Loss", ax=ax)
    if val_data is not None and not val_data.empty:
        sns.lineplot(data=val_data, x="epoch", y="val_loss", label="Val Loss", ax=ax)
    ax.set_title("Video-Level Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=600)
    plt.close(fig)


def plot_accuracy(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns:
        return
    train_data = df[["epoch", "train_acc"]].copy() if "train_acc" in df.columns else None
    val_data = df[["epoch", "val_acc"]].copy() if "val_acc" in df.columns else None
    if train_data is not None:
        train_data = train_data.apply(pd.to_numeric, errors="coerce").dropna()
    if val_data is not None:
        val_data = val_data.apply(pd.to_numeric, errors="coerce").dropna()
    if (train_data is None or train_data.empty) and (val_data is None or val_data.empty):
        logger.warning("Skipping accuracy plot: no accuracy data found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_data is not None and not train_data.empty:
        sns.lineplot(data=train_data, x="epoch", y="train_acc", label="Train Acc", ax=ax)
    if val_data is not None and not val_data.empty:
        sns.lineplot(data=val_data, x="epoch", y="val_acc", label="Val Acc", ax=ax)
    ax.set_title("Video-Level Accuracy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=600)
    plt.close(fig)


def plot_auc(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns or "val_auc" not in df.columns:
        return
    data = df[["epoch", "val_auc"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        logger.warning("Skipping AUC plot: no val_auc data found.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="epoch", y="val_auc", label="Val AUC", ax=ax)
    ax.set_title("Video-Level Validation AUC vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "val_auc.png", dpi=600)
    plt.close(fig)


def plot_learning_rate(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns or "learning_rate" not in df.columns:
        return
    data = df[["epoch", "learning_rate"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        logger.warning("Skipping learning rate plot: no learning_rate data found.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=data, x="epoch", y="learning_rate", label="Learning Rate", ax=ax)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate.png", dpi=600)
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
