import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from .config import (
    IMAGE_SIZE,
    CLASS_INDEX_PATH,
    MODEL_WEIGHTS_PATH,
)
from .model import get_model


# ---- Load class index ----
import json

with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    CLASS_INFO = json.load(f)

IDX_TO_CLASS = {int(k): v for k, v in CLASS_INFO["idx_to_class"].items()}
NUM_CLASSES = CLASS_INFO["num_classes"]


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_trained_model():
    """Load EfficientNet-B0 with trained weights."""
    model = get_model()   # already created with correct NUM_CLASSES
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def decode_label(label: str):
    """
    'rice_brownspot' -> crop='rice', disease='brownspot'
    'wheat_healthy'  -> crop='wheat', disease='healthy'
    """
    parts = label.split("_", maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


@torch.no_grad()
def predict_image(image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")

    transform = get_eval_transform()
    tensor = transform(img).unsqueeze(0)  # shape: (1, C, H, W)

    # Model
    model = load_trained_model()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tensor = tensor.to(device)

    # Forward
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)

    idx = pred_idx.item()
    confidence = conf.item()
    label = IDX_TO_CLASS[idx]

    crop, disease = decode_label(label)

    return {
        "class_label": label,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.inference <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    result = predict_image(img_path)

    print(f"Image: {img_path}")
    print(f"Predicted class: {result['class_label']}")
    print(f"Crop: {result['crop']}")
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
