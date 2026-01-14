import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PLANTDOC_TRAIN = PROJECT_ROOT / "data" / "raw" / "plantdoc" / "train"
RAW_PLANTDOC_TEST  = PROJECT_ROOT / "data" / "raw" / "plantdoc" / "test"

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
TRAIN_ROOT = PROCESSED_ROOT / "train"
VAL_ROOT   = PROCESSED_ROOT / "val"
TEST_ROOT  = PROCESSED_ROOT / "test"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_token(s: str) -> str:
    s = s.lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_ "
    s = "".join(ch if ch.isalnum() or ch in " _" else "_" for ch in s)
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def plantdoc_class_to_label(class_name: str) -> str:
    """
    Examples:
        'Apple leaf' -> apple_healthy
        'Apple rust leaf' -> apple_rust
        'Tomato Early blight leaf' -> tomato_early_blight
        'Tomato mold leaf' -> tomato_mold
    """
    name = class_name.strip()
    tokens = name.split()

    if not tokens:
        return "unknown_unknown"

    crop_tok = tokens[0]
    rest = tokens[1:]

    # remove trailing "leaf" / "leaves"
    if rest and rest[-1].lower() in ["leaf", "leaves"]:
        rest = rest[:-1]

    if not rest:
        disease = "healthy"
    else:
        disease = "_".join(normalize_token(t) for t in rest)

    crop = normalize_token(crop_tok)
    return f"{crop}_{disease}"


def copy_split(src_root: Path, dst_root: Path, split_name: str):
    ensure_dir(dst_root)
    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    print(f"{split_name}: found {len(class_dirs)} PlantDoc classes")

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = plantdoc_class_to_label(class_name)

        target_dir = dst_root / label
        ensure_dir(target_dir)

        for img_path in class_dir.iterdir():
            if img_path.is_file():
                dst_path = target_dir / img_path.name
                shutil.copy2(img_path, dst_path)

        print(f"{split_name}: {class_name} -> {label}")


def main():
    print("Preparing PlantDoc dataset...")
    print(f"RAW train: {RAW_PLANTDOC_TRAIN}")
    print(f"RAW test : {RAW_PLANTDOC_TEST}")
    print(f"OUT root : {PROCESSED_ROOT}")

    copy_split(RAW_PLANTDOC_TRAIN, TRAIN_ROOT, "train")
    # use PlantDoc test as our test set
    copy_split(RAW_PLANTDOC_TEST, TEST_ROOT, "test")

    print(" PlantDoc processing complete.")


if __name__ == "__main__":
    main()
