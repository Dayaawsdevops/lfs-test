"""
Image Classification Training Script
- Uses PyTorch + torchvision (ResNet18 transfer learning)
- Tracks experiments with MLflow
- Saves model to S3 via DVC
- Designed to run on AWS EC2 g4dn.xlarge or locally with CPU fallback
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR", "data/images")
MODEL_DIR       = Path("models")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "image-classification"

# Auto-detect GPU (NVIDIA on EC2) or fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────
def get_data_loaders(data_dir: str, batch_size: int = 32, img_size: int = 224):
    """
    Expects ImageFolder structure:
        data/images/
            train/
                class_a/  img1.jpg  img2.jpg ...
                class_b/  img1.jpg  img2.jpg ...
            val/          (optional — auto-split from train if missing)
                class_a/
                class_b/
    """
    # ImageNet-standard normalisation (works well for transfer learning)
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_path = Path(data_dir) / "train"
    val_path   = Path(data_dir) / "val"

    full_dataset = datasets.ImageFolder(str(train_path), transform=train_transforms)
    num_classes  = len(full_dataset.classes)
    class_names  = full_dataset.classes
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Total training images: {len(full_dataset)}")

    if val_path.exists():
        train_dataset = full_dataset
        val_dataset   = datasets.ImageFolder(str(val_path), transform=val_transforms)
    else:
        # Auto 80/20 split
        val_size   = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes, class_names


# ─────────────────────────────────────────────────────────────────
# Model: ResNet18 with Transfer Learning
# ─────────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = True):
    """
    ResNet18 pre-trained on ImageNet.
    We replace the final FC layer for our number of classes.
    freeze_backbone=True: only trains the final layer (fast, good for small datasets)
    freeze_backbone=False: fine-tunes all layers (better accuracy, needs more data/time)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────
def train(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
          freeze_backbone: bool = True, img_size: int = 224):

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        DATA_DIR, batch_size, img_size
    )

    model     = build_model(num_classes, freeze_backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    MODEL_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "freeze_backbone": freeze_backbone, "img_size": img_size,
            "num_classes": num_classes, "device": str(DEVICE),
            "model_arch": "resnet18"
        })

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_loss,   val_acc   = validate(model, val_loader, criterion)
            scheduler.step()

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss,     "val_acc": val_acc
            }, step=epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
                print(f"  -> New best model saved (val_acc={val_acc:.4f})")

        # Log final model to MLflow registry
        mlflow.pytorch.log_model(model, "model")

        # Save metadata
        metadata = {
            "val_acc":     round(best_val_acc, 4),
            "num_classes": num_classes,
            "class_names": class_names,
            "model_arch":  "resnet18",
            "img_size":    img_size
        }
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"))

        print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--img-size",        type=int,   default=224)
    parser.add_argument("--unfreeze",        action="store_true",
                        help="Fine-tune all layers instead of just final FC")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=not args.unfreeze,
        img_size=args.img_size
    )

