# scripts/test_r3d_forward_pass.py

import torch
import torchvision.models.video as models

# 1. Load model
model = models.r3d_18(pretrained=True)

# 2. Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# 3. Replace FC layer (7 classes)
model.fc = torch.nn.Linear(model.fc.in_features, 7)

# 4. Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 5. Create dummy input (batch_size=2, channels=3, frames=16, H=112, W=112)
dummy_input = torch.randn(2, 3, 16, 112, 112).to(device)

# 6. Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Should be (2, 7)
print(f"Model on device: {next(model.parameters()).device}")
print("✅ Forward pass successful!")
