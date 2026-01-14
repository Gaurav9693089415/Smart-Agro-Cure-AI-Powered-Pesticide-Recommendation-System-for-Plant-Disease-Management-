import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TRAIN = PROJECT_ROOT / "data" / "processed" / "train"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LABELS_PATH = ARTIFACTS_DIR / "class_index.json"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # all folders in train = classes
    class_dirs = sorted([d for d in PROCESSED_TRAIN.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    # build mapping: idx -> class_name
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}

    # optional: count images per class
    counts = {}
    for d in class_dirs:
        counts[d.name] = sum(1 for p in d.iterdir() if p.is_file())

    data = {
        "num_classes": len(class_names),
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "image_counts": counts,
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Found {len(class_names)} classes.")
    print(f"Saved class index to: {LABELS_PATH}")


if __name__ == "__main__":
    main()
