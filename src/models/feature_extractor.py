import torch
import torch.nn as nn
from src.models.r3d import create_r3d_classifier

class FeatureExtractor:
    """
    Wraps an R3D model to extract features instead of class scores.
    Replaces the final fully connected layer with Identity.
    """
    def __init__(self, checkpoint_path, num_classes=2, device='cuda'):
        self.device = device
        
        # 1. Instantiate the architecture
        # We perform "surgery" immediately after loading
        print(f"Loading base model from {checkpoint_path}...")
        self.model = create_r3d_classifier(num_classes=num_classes, pretrained=False)
        
        # 2. Load Weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle potential prefix issues (e.g. 'module.')
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)
        
        # 3. Remove Classification Head
        # R3D-18 uses 'fc' as the final layer. We replace it.
        # This turns the (Batch, 512) -> (Batch, 2) into (Batch, 512) -> (Batch, 512)
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        else:
            print("Warning: Model has no 'fc' layer. Check architecture.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, clip_tensor):
        """
        Input: Tensor (Batch, 3, 16, H, W)
        Output: Numpy Array (Batch, 512)
        """
        clip_tensor = clip_tensor.to(self.device)
        
        # Forward pass (now stops before classification)
        features = self.model(clip_tensor)
        
        # Flatten if necessary (Batch, 512, 1, 1, 1) -> (Batch, 512)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy()