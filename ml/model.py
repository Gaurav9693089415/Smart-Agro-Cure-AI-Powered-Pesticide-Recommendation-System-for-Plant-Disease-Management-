from torch import nn
import torch
from torchvision import models

from .config import NUM_CLASSES, MODEL_NAME, DEVICE


def get_model():
    """
    Create an EfficientNet-B0 model with ImageNet weights,
    and replace the classifier head for NUM_CLASSES.
    """
    if MODEL_NAME == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features

        # replace classifier
        backbone.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        model = backbone

    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(model)
    print(f"Trainable params: {count_parameters(model):,}")
    print(f"Device: {next(model.parameters()).device}")
