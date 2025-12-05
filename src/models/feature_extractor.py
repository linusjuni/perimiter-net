import torch
from src.models.r3d import create_r3d_classifier


class FeatureExtractor:
    """R3D feature extractor wrapper."""

    def __init__(self, checkpoint_path, num_classes=2, device="cuda"):
        self.device = device
        print(f"Loading model from {checkpoint_path}...")
        self.wrapper = create_r3d_classifier(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.wrapper.load_state_dict(clean_state_dict, strict=True)
        self.model = self.wrapper.model
        self.model.fc = None
        self.wrapper.to(self.device)
        self.wrapper.eval()
        print("✓ Model loaded successfully")

    def _forward_features(self, x):
        """Extract 512D features from R3D backbone."""
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)
        return x

    @torch.no_grad()
    def extract(self, clip_tensor):
        """Extract features from video clips."""
        clip_tensor = clip_tensor.to(self.device)
        features = self._forward_features(clip_tensor)
        return features.cpu().numpy()
