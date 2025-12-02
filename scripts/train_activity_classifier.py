"""Train the Simple3DCNN on VIRAT clips.

Just press “Run Python File” in your IDE; tweak the constants below if needed.
"""
import os
import sys

import torch

# Make src importable when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training import ActivityTrainer, TrainerConfig  # noqa: E402

# Default settings you can edit directly
DATA_ROOT = os.environ.get("VIRAT_DATA_ROOT", "data")
EPOCHS = 5
BATCH_SIZE = 4
CLIP_LEN = 16
RESIZE = (112, 112)  # set to None to keep original size
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "artifacts/activity3d.pt"


def main():
    config = TrainerConfig(
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        clip_len=CLIP_LEN,
        resize=RESIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        save_path=SAVE_PATH,
    )

    print("TrainerConfig:", config)
    trainer = ActivityTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
