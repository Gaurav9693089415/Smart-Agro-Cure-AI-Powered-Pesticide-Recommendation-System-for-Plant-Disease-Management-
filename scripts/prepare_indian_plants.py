import shutil
import random
from pathlib import Path

# --------- CONFIG ---------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_INDIAN = PROJECT_ROOT / "data" / "raw" / "indian_plants"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

TRAIN_ROOT = PROCESSED_ROOT / "train"
VAL_ROOT   = PROCESSED_ROOT / "val"

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
# --------------------------
#  Only these 4 crops will be processed
ALLOWED_CROPS = {"Rice", "Wheat", "Corn", "Cotton"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_token(s: str) -> str:
    s = s.lower().strip()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_ "
    s = "".join(ch if ch.isalnum() or ch in " _" else "_" for ch in s)
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def indian_class_to_label(crop_name: str, disease_name: str) -> str:
    """
    Examples:
        crop_name   = 'Rice', disease_name = 'Brownspot'          -> rice_brownspot
        crop_name   = 'Coconut', disease_name = 'CCI_Caterpillars' -> coconut_cci_caterpillars
    """
    crop = normalize_token(crop_name)
    disease = normalize_token(disease_name)
    return f"{crop}_{disease}"


def collect_image_dirs(disease_root: Path):
    """
    In your dataset, structure looks like:
        indian_plants/Coconut/CCI_Caterpillars/CCI_Caterpillars/*.jpg
    So sometimes there is an extra nested folder.
    This function returns the deepest folders that contain images.
    """
    subdirs = [d for d in disease_root.iterdir() if d.is_dir()]
    if not subdirs:
        return [disease_root]
    # if inside there are further dirs, use those
    leaf_dirs = []
    for d in subdirs:
        inner = [x for x in d.iterdir() if x.is_dir()]
        if inner:
            leaf_dirs.extend(inner)
        else:
            leaf_dirs.append(d)
    return leaf_dirs


def process_indian():
    random.seed(RANDOM_SEED)

    ensure_dir(TRAIN_ROOT)
    ensure_dir(VAL_ROOT)

    crop_dirs = [d for d in RAW_INDIAN.iterdir() if d.is_dir()]
    print(f"Found {len(crop_dirs)} Indian crops in raw dataset")

    for crop_dir in crop_dirs:
        crop_name = crop_dir.name  # e.g. 'Rice', 'Wheat', 'Coconut'

        #  Skip everything except Rice, Wheat, Corn, Cotton
        if crop_name not in ALLOWED_CROPS:
            print(f"Skipping crop: {crop_name}")
            continue

        print(f"Processing crop: {crop_name}")

        disease_dirs = [d for d in crop_dir.iterdir() if d.is_dir()]
        if not disease_dirs:
            continue

        for disease_dir in disease_dirs:
            disease_name = disease_dir.name  # e.g. 'Brownspot', 'CCI_Caterpillars'

            leaf_dirs = collect_image_dirs(disease_dir)

            images = []
            for ld in leaf_dirs:
                for img_path in ld.iterdir():
                    if img_path.is_file():
                        images.append(img_path)

            if not images:
                continue

            random.shuffle(images)
            split_idx = int(len(images) * TRAIN_SPLIT)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            label = indian_class_to_label(crop_name, disease_name)

            train_target = TRAIN_ROOT / label
            val_target = VAL_ROOT / label
            ensure_dir(train_target)
            ensure_dir(val_target)

            for img_path in train_imgs:
                dst = train_target / img_path.name
                shutil.copy2(img_path, dst)

            for img_path in val_imgs:
                dst = val_target / img_path.name
                shutil.copy2(img_path, dst)

            print(
                f"{crop_name}/{disease_name} -> {label}: "
                f"{len(train_imgs)} train, {len(val_imgs)} val"
            )

    print(" Indian plants processing complete (Rice, Wheat, Corn, Cotton only).")
    

if __name__ == "__main__":
    print("Preparing indian_plants dataset...")
    print(f"RAW: {RAW_INDIAN}")
    print(f"OUT: {PROCESSED_ROOT}")
    process_indian()
