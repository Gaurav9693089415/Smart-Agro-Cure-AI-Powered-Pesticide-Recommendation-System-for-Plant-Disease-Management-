from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)


def get_transforms():
    """Transforms for train / val / test."""
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, eval_tfms


def get_dataloaders():
    train_tfms, eval_tfms = get_transforms()

    train_ds = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=str(VAL_DIR),   transform=eval_tfms)

    # test dir may be empty if only PlantDoc added there — handle both cases
    if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
        test_ds = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_tfms)
    else:
        test_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
