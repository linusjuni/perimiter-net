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

TRAIN_COMPONENTS = [
    ("train_rank_loss", "Rank Loss"),
    ("train_sparsity_loss", "Sparsity Loss"),
    ("train_smoothness_loss", "Smoothness Loss"),
]

VAL_COMPONENTS = [
    ("val_rank_loss", "Rank Loss"),
    ("val_sparsity_loss", "Sparsity Loss"),
    ("val_smoothness_loss", "Smoothness Loss"),
]


def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"training_history.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_out_dir(run_name: str) -> Path:
    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / "plots" / "mil" / run_name
    return out_dir


def _prepare_component_frame(df: pd.DataFrame, components):
    available = [c for c, _ in components if c in df.columns]
    if "epoch" not in df.columns or not available:
        return pd.DataFrame()

    data = df[["epoch"] + available].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=["epoch"])
    melted = data.melt(
        id_vars="epoch",
        value_vars=available,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    label_map = {col: label for col, label in components if col in df.columns}
    melted["metric"] = melted["metric"].map(label_map)
    return melted


def plot_total_loss(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns:
        logger.warning("Skipping total loss plot: 'epoch' column missing.")
        return

    train_data = None
    val_data = None
    if "train_loss" in df.columns:
        train_data = df[["epoch", "train_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    if "val_loss" in df.columns:
        val_data = df[["epoch", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()

    if (train_data is None or train_data.empty) and (val_data is None or val_data.empty):
        logger.warning("Skipping total loss plot: no loss columns found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_data is not None and not train_data.empty:
        sns.lineplot(data=train_data, x="epoch", y="train_loss", label="Train Loss", ax=ax)
    if val_data is not None and not val_data.empty:
        sns.lineplot(data=val_data, x="epoch", y="val_loss", label="Val Loss", ax=ax)

    ax.set_title("MIL Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)


def plot_loss_components(df: pd.DataFrame, out_dir: Path, split: str):
    components = TRAIN_COMPONENTS if split == "train" else VAL_COMPONENTS
    melted = _prepare_component_frame(df, components)
    if melted.empty:
        logger.warning(f"Skipping {split} loss components plot: missing data.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=melted, x="epoch", y="value", hue="metric", ax=ax)
    ax.set_title(f"MIL {split.title()} Loss Components vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(title=None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_loss_components.png", dpi=150)
    plt.close(fig)


def plot_auc(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns or "val_auc" not in df.columns:
        logger.warning("Skipping AUC plot: required columns missing.")
        return

    data = df[["epoch", "val_auc"]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        logger.warning("Skipping AUC plot: no val_auc data found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="epoch", y="val_auc", label="Val AUC", ax=ax)
    ax.set_title("MIL Validation AUC vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "val_auc.png", dpi=150)
    plt.close(fig)


def plot_learning_rate(df: pd.DataFrame, out_dir: Path):
    if "epoch" not in df.columns or "learning_rate" not in df.columns:
        logger.warning("Skipping learning rate plot: required columns missing.")
        return

    data = df[["epoch", "learning_rate"]].apply(pd.to_numeric, errors="coerce").dropna()
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
    fig.savefig(out_dir / "learning_rate.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.plotting.plot_mil_training_history <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    csv_path = run_dir / "training_history.csv"

    try:
        df = load_history(csv_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_name = run_dir.name
    out_dir = ensure_out_dir(run_name)
    logger.info(f"Loaded MIL history from {csv_path}")
    logger.info(f"Saving plots to: {out_dir}")

    plot_total_loss(df, out_dir)
    plot_loss_components(df, out_dir, split="train")
    plot_loss_components(df, out_dir, split="val")
    plot_auc(df, out_dir)
    plot_learning_rate(df, out_dir)

    logger.info("Finished plotting MIL training history.")


if __name__ == "__main__":
    main()
