from pathlib import Path
import json
from typing import Dict

import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# This file: backend/app/ml/inference.py
# parents[0] = ml, [1] = app, [2] = backend, [3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASS_INDEX_PATH = ARTIFACTS_DIR / "class_index.json"
MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / "model_efficientnet_b0.pth"

with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    CLASS_INFO = json.load(f)

NUM_CLASSES = CLASS_INFO["num_classes"]
IDX_TO_CLASS = CLASS_INFO["idx_to_class"]  # keys are strings: "0", "1", ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Build model same as in training ----
model = efficientnet_b0(weights=None)
# EfficientNet-B0 classifier is Sequential[Dropout, Linear]
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _split_class_name(class_name: str):
    """
    'rice_bacterialblight' -> ('rice', 'bacterialblight')
    'wheat_stripe_rust'   -> ('wheat', 'stripe rust')
    """
    parts = class_name.split("_", 1)
    crop = parts[0]
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "healthy"
    return crop, disease


def predict_image(image_path: str | Path) -> Dict:
    """
    Run EfficientNet model on a single image and return structured prediction.
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    pred_class = IDX_TO_CLASS[str(pred_idx)]

    crop, disease = _split_class_name(pred_class)

    return {
        "pred_class": pred_class,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
    }
