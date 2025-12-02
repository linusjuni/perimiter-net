import torch
import torch.nn as nn
from src.models.r3d import create_r3d_classifier

class FeatureExtractor:
    """
    Wraps an R3D model to extract features instead of class scores.
    """
    def __init__(self, checkpoint_path, num_classes=2, device='cuda'):
        self.device = device
        
        # 1. Instantiate Architecture
        print(f"Loading base model from {checkpoint_path}...")
        # pretrained=False because we load our own weights immediately after
        self.model = create_r3d_classifier(num_classes=num_classes, pretrained=False)
        
        # 2. Load Weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle 'module.' prefix if saved from DataParallel
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)
        
        # 3. Surgery: Replace FC with Identity
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, clip_tensor):
        clip_tensor = clip_tensor.to(self.device)
        
        # Forward pass
        # R3D: input -> avgpool -> flatten -> fc(Identity)
        features = self.model(clip_tensor)
        
        # Ensure flat vector (Batch, 512)
        # Addressing Critique #1: R3D avgpool returns (B, 512, 1, 1, 1)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy()