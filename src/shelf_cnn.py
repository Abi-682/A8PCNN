#!/usr/bin/env python3
"""Shelf CNN assignment implementation.

This script builds, trains, and evaluates a CNN for the synthetic shelf inspection dataset.
It also supports dataset generation, early stopping, full regularization, and filter visualization.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
import matplotlib.pyplot as plt


DATA_FILENAME = "shelf_images.npz"
CLASS_NAMES = ["normal", "damaged", "overloaded"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_shelf_image(label: int, image_size: int = 64, noise_std: float = 0.08) -> np.ndarray:
    """Generate one synthetic shelf image for the given label."""
    rng = np.random.default_rng()
    canvas = np.ones((image_size, image_size), dtype=np.float32)

    # Shelf bar properties.
    bar_height = 6
    shelf_y = image_size // 2 + rng.integers(-4, 5)
    shelf_y = np.clip(shelf_y, 16, image_size - 20)
    bar_brightness = 0.45 + rng.uniform(-0.08, 0.08)
    canvas[shelf_y : shelf_y + bar_height, :] = bar_brightness

    def draw_boxes(count: int, min_width: int, max_width: int, min_height: int, max_height: int, density: float = 0.7) -> None:
        x = 4
        while x < image_size - 12 and count > 0:
            width = rng.integers(min_width, max_width + 1)
            height = rng.integers(min_height, max_height + 1)
            y = shelf_y - height
            x += rng.integers(0, 4)
            x_end = min(image_size - 4, x + width)
            if x_end - x < 8:
                break
            color = 0.20 + rng.uniform(-0.05, 0.10)
            canvas[max(0, y) : shelf_y - 2, x:x_end] = np.clip(color + rng.normal(0, 0.03), 0.0, 1.0)
            x = x_end + rng.integers(2, 6)
            count -= 1

    if label == 0:
        draw_boxes(count=4, min_width=12, max_width=18, min_height=14, max_height=22)
    elif label == 1:
        draw_boxes(count=3, min_width=12, max_width=18, min_height=14, max_height=22)
        crack_x = rng.integers(12, image_size - 12)
        crack_y1 = shelf_y + 1
        crack_y2 = min(image_size - 1, shelf_y + rng.integers(12, 22))
        for dy in range(crack_y2 - crack_y1):
            x = crack_x + rng.integers(-1, 2)
            y = crack_y1 + dy
            if 0 <= x < image_size and 0 <= y < image_size:
                canvas[y, x] = 0.0
        if rng.random() < 0.5:
            damaged_box_x = rng.integers(16, image_size - 20)
            damaged_box_w = rng.integers(10, 16)
            damaged_box_h = rng.integers(10, 18)
            damaged_box_y = shelf_y - damaged_box_h
            canvas[damaged_box_y:shelf_y - 2, damaged_box_x:damaged_box_x + damaged_box_w] = 0.0
    else:
        draw_boxes(count=6, min_width=10, max_width=16, min_height=18, max_height=28)
        overfill = rng.integers(2, 4)
        for _ in range(overfill):
            box_w = rng.integers(8, 14)
            box_h = rng.integers(10, 18)
            x = rng.integers(4, image_size - box_w - 4)
            y = rng.integers(max(2, shelf_y - box_h - 8), shelf_y - 6)
            color = 0.18 + rng.uniform(-0.05, 0.05)
            canvas[y : y + box_h, x : x + box_w] = np.clip(color + rng.normal(0, 0.03), 0.0, 1.0)

    noise = rng.normal(0.0, noise_std, size=canvas.shape).astype(np.float32)
    image = np.clip(canvas + noise, 0.0, 1.0)
    return image


def generate_shelf_dataset(path: Path, n_per_class: int = 300, seed: int = 42) -> None:
    """Generate and save the synthetic shelf dataset."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n_per_class * len(CLASS_NAMES), 64, 64), dtype=np.float32)
    labels = np.zeros((n_per_class * len(CLASS_NAMES),), dtype=np.int64)

    for label in range(len(CLASS_NAMES)):
        for idx in range(n_per_class):
            i = label * n_per_class + idx
            images[i] = generate_shelf_image(label)
            labels[i] = label

    save_path = path / DATA_FILENAME
    np.savez_compressed(save_path, images=images, labels=labels, class_names=np.array(CLASS_NAMES, dtype=object))
    print(f"Saved synthetic dataset to {save_path}")


class ShelfImageDataset(Dataset):
    """PyTorch dataset for the synthetic shelf images."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = int(self.labels[index])
        image = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ShelfCNN(nn.Module):
    def __init__(self, use_batchnorm: bool = True, dropout_p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResNetTransfer(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if self.model.conv1.in_channels != 1:
            weight = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight.data = weight.mean(dim=1, keepdim=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def split_indices(n: int, train_frac: float, val_frac: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def make_dataloaders(images: np.ndarray, labels: np.ndarray, batch_size: int, augment: bool, transfer_learning: bool):
    train_transform = [transforms.ToTensor()]
    if augment:
        train_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10, fill=255),
            transforms.ColorJitter(brightness=0.25),
            transforms.ToTensor(),
        ]
    val_transform = [transforms.ToTensor()]

    if transfer_learning:
        normalization = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalization = transforms.Normalize(mean=[0.5], std=[0.5])

    train_transform.append(normalization)
    val_transform.append(normalization)

    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)

    train_idx, val_idx, test_idx = split_indices(len(labels), train_frac=0.7, val_frac=0.15)
    train_set = Subset(ShelfImageDataset(images, labels, transform=train_transform), train_idx)
    val_set = Subset(ShelfImageDataset(images, labels, transform=val_transform), val_idx)
    test_set = Subset(ShelfImageDataset(images, labels, transform=val_transform), test_idx)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True),
        train_idx,
        val_idx,
        test_idx,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            losses.append(loss.item() * X.size(0))
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += X.size(0)
    return sum(losses) / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    output_dir: Path,
) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Epoch 0 evaluation before training.
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(float("nan"))
    history["train_acc"].append(float("nan"))
    print(f"Epoch 0 (untrained)  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * X.size(0)
            running_correct += (preds == y).sum().item()
            running_total += X.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print("  New best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Stopping early after {epoch} epochs (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best_model.pth")
        print(f"Best model weights saved to {output_dir / 'best_model.pth'}")

    return history


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_history(history: dict, output_path: Path) -> None:
    epochs = list(range(len(history["val_loss"])))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], label="Train loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_acc"], label="Train acc", linestyle="--", marker="x")
    ax2.plot(epochs, history["val_acc"], label="Val acc", linestyle="--", marker="x")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved training plot to {output_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def save_example_predictions(
    model: nn.Module,
    dataset: Subset,
    device: torch.device,
    output_path: Path,
    n_correct: int = 5,
    n_incorrect: int = 5,
) -> None:
    model.eval()
    examples = []
    with torch.no_grad():
        for X, y in DataLoader(dataset, batch_size=32, shuffle=False):
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.numpy()
            for i in range(X.size(0)):
                image = X[i].cpu().squeeze(0).numpy()
                label = int(y_np[i])
                pred = int(preds[i])
                examples.append((image, label, pred))

    correct = [e for e in examples if e[1] == e[2]][:n_correct]
    incorrect = [e for e in examples if e[1] != e[2]][:n_incorrect]
    selected = correct + incorrect
    n = len(selected)
    if n == 0:
        return

    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for idx, (image, label, pred) in enumerate(selected):
        ax = axes[idx]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        title = f"T={CLASS_NAMES[label]}\nP={CLASS_NAMES[pred]}"
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved example predictions to {output_path}")


def visualize_first_layer_filters(model: ShelfCNN, output_path: Path) -> None:
    weights = model.features[0].weight.detach().cpu().numpy()
    n_filters = weights.shape[0]
    cols = 4
    rows = math.ceil(n_filters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)
    for idx in range(len(axes)):
        ax = axes[idx]
        ax.axis("off")
        if idx < n_filters:
            filt = weights[idx, 0]
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            ax.imshow(filt, cmap="gray")
    fig.suptitle("First convolutional layer filters")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved filter visualizations to {output_path}")


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    file_path = root / DATA_FILENAME
    if not file_path.exists():
        print(f"Dataset file {file_path} not found. Generating synthetic dataset.")
        generate_shelf_dataset(root)
    data = np.load(file_path, allow_pickle=True)
    images = data["images"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    class_names = list(data["class_names"].tolist())
    return images, labels, class_names


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a shelf inspection CNN.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root directory for shelf_images.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for saved plots and model weights")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability after the fully connected layer")
    parser.add_argument("--no-batchnorm", action="store_true", help="Disable batch normalization")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training data augmentation")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--transfer-learning", action="store_true", help="Use pretrained ResNet-18 instead of the custom CNN")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained ResNet backbone when using transfer learning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split and generation")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    images, labels, class_names = load_dataset(root)
    if class_names != CLASS_NAMES:
        print("Warning: class names from dataset differ from expected assignment classes.")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = make_dataloaders(
        images,
        labels,
        batch_size=args.batch_size,
        augment=not args.no_augmentation,
        transfer_learning=args.transfer_learning,
    )

    if args.transfer_learning:
        model = ResNetTransfer(pretrained=True, freeze_backbone=args.freeze_backbone)
    else:
        model = ShelfCNN(use_batchnorm=not args.no_batchnorm, dropout_p=args.dropout)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        output_dir=output_dir,
    )

    plot_history(history, output_dir / "training_history.png")

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"Test set: loss={test_loss:.4f} accuracy={test_acc:.4f}")

    preds, targets = predict_dataset(model, test_loader, device)
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    print("\nClassification report:\n", report)
    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    test_examples = Subset(ShelfImageDataset(images, labels, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])), test_idx)
    save_example_predictions(model, test_examples, device, output_dir / "example_predictions.png")

    if not args.transfer_learning:
        visualize_first_layer_filters(model, output_dir / "first_layer_filters.png")

    print("Done. Outputs are available in:", output_dir)


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
