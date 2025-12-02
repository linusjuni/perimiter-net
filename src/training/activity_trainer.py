from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.virat import VIRATDataset
from src.models.activity import Simple3DCNN


def build_default_transform(resize: Optional[Tuple[int, int]] = (112, 112)):
    """Convert numpy frames (T, H, W, C) to torch (C, T, H, W); optional resize."""

    def _transform(frames):
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        if resize is not None:
            tensor = F.interpolate(
                tensor.permute(1, 0, 2, 3),  # (T, C, H, W)
                size=resize,
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)  # back to (C, T, H, W)
        return tensor

    return _transform


@dataclass
class TrainerConfig:
    data_root: str
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_len: int = 16
    resize: Optional[Tuple[int, int]] = (112, 112)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    save_path: Optional[str] = "artifacts/activity3d.pt"


class ActivityTrainer:
    """Minimal training/eval loop for the 3D activity model."""

    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.device = config.device

        transform = build_default_transform(config.resize)
        self.train_ds = VIRATDataset(
            root_dir=config.data_root, split="train", clip_len=config.clip_len, transform=transform
        )
        self.val_ds = VIRATDataset(
            root_dir=config.data_root, split="val", clip_len=config.clip_len, transform=transform
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda",
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda",
        )

        self.model = Simple3DCNN(num_classes=len(VIRATDataset.ACTIVITIES))
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        if config.save_path:
            Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for batch_idx, (clips, labels) in enumerate(self.train_loader, start=1):
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(clips)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

            if batch_idx % 10 == 0:
                avg_loss = total_loss / total if total else 0.0
                avg_acc = total_correct / total if total else 0.0
                print(f"[Epoch {epoch}] Step {batch_idx}/{len(self.train_loader)} - loss {avg_loss:.4f}, acc {avg_acc:.3f}")

        return (
            total_loss / total if total else 0.0,
            total_correct / total if total else 0.0,
        )

    @torch.no_grad()
    def evaluate(self):
        if len(self.val_ds) == 0:
            print("No val samples; skipping eval.")
            return 0.0, 0.0

        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0

        for clips, labels in self.val_loader:
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(clips)
            loss = self.criterion(logits, labels)

            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

        return (
            total_loss / total if total else 0.0,
            total_correct / total if total else 0.0,
        )

    def fit(self):
        if len(self.train_ds) == 0:
            print("No training samples found. Check data_root and splits.")
            return

        best_acc = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            if self.cfg.save_path and val_acc >= best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.cfg.save_path)
                print(f"Saved checkpoint to {self.cfg.save_path}")
