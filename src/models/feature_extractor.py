import torch
from src.models.r3d import create_r3d_classifier


class FeatureExtractor:
    """
    Wraps an R3D model to extract 512D features from the backbone.
    Removes the classification head to extract pre-logit features.
    """

    def __init__(self, checkpoint_path, num_classes=2, device="cuda"):
        self.device = device

        # Load trained R3D model
        print(f"Loading model from {checkpoint_path}...")
        self.wrapper = create_r3d_classifier(num_classes=num_classes, pretrained=False)

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Handle DataParallel 'module.' prefix
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.wrapper.load_state_dict(clean_state_dict, strict=True)

        # Access the actual R3D model and remove FC layer
        self.model = self.wrapper.model
        self.model.fc = None

        self.wrapper.to(self.device)
        self.wrapper.eval()
        print("✓ Model loaded successfully")

    def _forward_features(self, x):
        """Extract 512D features from R3D backbone (before FC layer)."""
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)  # (B, 512, 1, 1, 1) -> (B, 512)
        return x

    @torch.no_grad()
    def extract(self, clip_tensor):
        """
        Extract features from video clips.

        Args:
            clip_tensor (torch.Tensor): Shape (B, 3, 16, H, W)

        Returns:
            numpy.ndarray: Shape (B, 512)
        """
        clip_tensor = clip_tensor.to(self.device)
        features = self._forward_features(clip_tensor)
        return features.cpu().numpy()
