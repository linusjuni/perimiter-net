import torch
from src.utils.training_utils import create_r3d_classifier


def main():
    # Instantiate the model
    num_classes = 7
    model = create_r3d_classifier(
        num_classes=num_classes, pretrained=True, freeze_backbone=True, dropout=0.5
    )

    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # Dummy input: (batch_size, channels, time, height, width)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")  # Should be (2, num_classes)


if __name__ == "__main__":
    main()
