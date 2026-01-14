import time

import torch
from torch import nn, optim

from ml.config import (
    ARTIFACTS_DIR,
    MODEL_WEIGHTS_PATH,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from ml.data_module import get_dataloaders
from ml.model import get_model, count_parameters


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if loader is None:
        return None, None

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = get_model().to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_weights_path = MODEL_WEIGHTS_PATH

    # ---- Early stopping settings ----
    patience = 3          # stop if no improvement for these many epochs
    min_delta = 1e-4      # minimum improvement to count
    no_improve_epochs = 0
    # ---------------------------------

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
            f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f} "
            f"- Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc is not None and (val_acc - best_val_acc) > min_delta:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f" New best model saved with val_acc={best_val_acc:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        # Early stopping check
        if no_improve_epochs >= patience:
            print(" Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best weights stored at: {best_weights_path}")

    # Optional: evaluate on test set if available
    if test_loader is not None:
        print("Evaluating on test set...")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
