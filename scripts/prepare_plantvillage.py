import os
import shutil
import random
from pathlib import Path

# --------- CONFIG ---------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PLANTVILLAGE = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "color"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

TRAIN_SPLIT = 0.8   # 80% train, 20% val (we'll create test later)
RANDOM_SEED = 42
# --------------------------


def clean_name(name: str) -> str:
    """
    Convert raw folder/class name to crop_disease format.
    Examples:
        'Corn_(maize)___Northern_Leaf_Blight' -> 'corn_northern_leaf_blight'
        'Potato___healthy' -> 'potato_healthy'
    """
    # PlantVillage uses 'Crop___Disease'
    if "___" in name:
        crop, disease = name.split("___", maxsplit=1)
    else:
        # fallback (shouldn't happen here)
        parts = name.split("__", maxsplit=1)
        crop = parts[0]
        disease = parts[1] if len(parts) > 1 else "unknown"

    # remove brackets etc.
    def normalize(s: str) -> str:
        s = s.lower()
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_ "
        s = "".join(ch if ch.isalnum() or ch in " _" else "_" for ch in s)
        s = s.replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    crop_clean = normalize(crop)
    disease_clean = normalize(disease)
    return f"{crop_clean}_{disease_clean}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_plantvillage():
    random.seed(RANDOM_SEED)

    train_root = PROCESSED_ROOT / "train"
    val_root = PROCESSED_ROOT / "val"

    ensure_dir(train_root)
    ensure_dir(val_root)

    class_dirs = [d for d in RAW_PLANTVILLAGE.iterdir() if d.is_dir()]

    print(f"Found {len(class_dirs)} PlantVillage classes")

    for class_dir in class_dirs:
        raw_class_name = class_dir.name
        target_class = clean_name(raw_class_name)

        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        train_target_dir = train_root / target_class
        val_target_dir = val_root / target_class
        ensure_dir(train_target_dir)
        ensure_dir(val_target_dir)

        for img_path in train_imgs:
            dest = train_target_dir / img_path.name
            shutil.copy2(img_path, dest)

        for img_path in val_imgs:
            dest = val_target_dir / img_path.name
            shutil.copy2(img_path, dest)

        print(
            f"{raw_class_name} -> {target_class}: "
            f"{len(train_imgs)} train, {len(val_imgs)} val"
        )

    print(" PlantVillage processing complete.")


if __name__ == "__main__":
    print("Preparing PlantVillage dataset...")
    print(f"RAW: {RAW_PLANTVILLAGE}")
    print(f"OUT: {PROCESSED_ROOT}")
    process_plantvillage()
