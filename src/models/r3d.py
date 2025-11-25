import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import logging


class R3DClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.5,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing R3DClassifier...")

        # Load R3D-18 model
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = r3d_18(weights=weights)

        # Replace the final fully connected layer with dropout + linear
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
            self.logger.info(
                "Backbone frozen (stem + layer1-4). Only FC layer is trainable."
            )

        # Log parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        return self.model(x)


def create_r3d_classifier(num_classes=7, **kwargs):
    """Factory function for easy model instantiation."""
    return R3DClassifier(num_classes=num_classes, **kwargs)
