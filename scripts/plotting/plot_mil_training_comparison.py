import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")
sns.set_palette("muted")

BEST_LINE_COLOR = "red"


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


def build_loss_frame(histories: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for run_name, df in histories:
        if "epoch" not in df.columns:
            continue
        for split, col in (("Train", "train_loss"), ("Val", "val_loss")):
            if col not in df.columns:
                continue
            data = df[["epoch", col]].apply(pd.to_numeric, errors="coerce").dropna()
            data = data.rename(columns={col: "value"})
            data["split"] = split
            data["run"] = run_name
            rows.append(data)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_auc_frame(histories: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for run_name, df in histories:
        if "epoch" not in df.columns or "val_auc" not in df.columns:
            continue
        data = df[["epoch", "val_auc"]].apply(pd.to_numeric, errors="coerce").dropna()
        data = data.rename(columns={"val_auc": "value"})
        data["run"] = run_name
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def best_val_loss_epochs(histories: List[Tuple[str, pd.DataFrame]]) -> Dict[str, float]:
    best_epochs: Dict[str, float] = {}
    for run_name, df in histories:
        if "epoch" not in df.columns or "val_loss" not in df.columns:
            continue
        val_df = df[["epoch", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
        if val_df.empty:
            continue
        min_idx = val_df["val_loss"].idxmin()
        best_epochs[run_name] = float(val_df.loc[min_idx, "epoch"])
    return best_epochs


def plot_comparison(histories: List[Tuple[str, pd.DataFrame]], out_dir: Path, tag: str):
    loss_df = build_loss_frame(histories)
    auc_df = build_auc_frame(histories)
    best_epochs = best_val_loss_epochs(histories)

    if loss_df.empty and auc_df.empty:
        logger.error("No comparable metrics found across runs.")
        return

    run_names = [name for name, _ in histories]
    palette_colors = sns.color_palette("muted", n_colors=len(run_names))
    palette = {run: color for run, color in zip(run_names, palette_colors)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    if not loss_df.empty:
        sns.lineplot(
            data=loss_df, x="epoch", y="value", hue="run", style="split", palette=palette, ax=axes[0]
        )
        axes[0].set_title("Train/Val Loss Comparison")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        y_top = axes[0].get_ylim()[1]
        for run, epoch in best_epochs.items():
            axes[0].axvline(epoch, color=BEST_LINE_COLOR, linestyle="--", linewidth=1.1, alpha=0.9)
            axes[0].text(
                epoch,
                y_top,
                f"best {run}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8.5,
                color=BEST_LINE_COLOR,
            )
    else:
        axes[0].set_visible(False)

    # AUC subplot
    if not auc_df.empty:
        sns.lineplot(data=auc_df, x="epoch", y="value", hue="run", palette=palette, ax=axes[1])
        axes[1].set_title("Validation AUC Comparison")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / f"{tag}_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MIL training histories from multiple run directories in a single plot."
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path, help="Paths to run directories containing training_history.csv"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for output folder/file name (defaults to run names joined).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs: List[Path] = args.run_dirs
    tag = args.tag

    histories: List[Tuple[str, pd.DataFrame]] = []
    missing = []
    for run_dir in run_dirs:
        try:
            df = load_history(run_dir)
            histories.append((run_dir.name, df))
        except FileNotFoundError:
            missing.append(str(run_dir))

    if missing:
        logger.warning(f"Skipped missing histories: {missing}")
    if not histories:
        logger.error("No valid training histories to plot.")
        sys.exit(1)

    if tag is None:
        tag = "__".join([name for name, _ in histories])

    out_dir = ensure_out_dir(tag)
    logger.info(f"Saving comparison plots to: {out_dir}")
    plot_comparison(histories, out_dir, tag)


if __name__ == "__main__":
    main()
