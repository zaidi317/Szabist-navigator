"""
Fine-tune DINOv2 for campus location classification.

This script is designed for adding more classes easily:
- Put new classes as new folders under dataset/train and dataset/test
- Run this script to fine-tune a classifier

Outputs:
- checkpoints/best_finetuned_dinov2.pth
- checkpoints/class_to_idx.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BASE_DIR, MODEL_NAME


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear classifier head."""

    def __init__(self, num_classes: int, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # Infer embedding dim from model name, default fallback for vits14.
        if "vits14" in model_name:
            embed_dim = 384
        elif "vitb14" in model_name:
            embed_dim = 768
        elif "vitl14" in model_name:
            embed_dim = 1024
        elif "vitg14" in model_name:
            embed_dim = 1536
        else:
            embed_dim = 384

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def build_dataloaders(
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train/test dataloaders with standard ImageNet normalization."""
    train_tfms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    test_tfms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=test_tfms)

    if train_ds.class_to_idx != test_ds.class_to_idx:
        logger.warning(
            "Train/Test class mapping differs. Ensure both contain same class folders."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_ds.class_to_idx


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (loss, accuracy_percent)."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 classifier")
    parser.add_argument("--train-dir", type=str, default=str(BASE_DIR / "dataset" / "train"))
    parser.add_argument("--test-dir", type=str, default=str(BASE_DIR / "dataset" / "test"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="If set, train backbone with smaller LR. Otherwise only classifier head is trained.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default=str(BASE_DIR / "checkpoints"))
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Train/Test directory missing: {train_dir} | {test_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    logger.info("Detected %d classes: %s", num_classes, list(class_to_idx.keys()))

    model = DINOv2Classifier(num_classes=num_classes, model_name=args.model_name).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = bool(args.unfreeze_backbone)

    # Parameter groups with different LRs.
    param_groups = [{"params": model.classifier.parameters(), "lr": args.lr_head}]
    if args.unfreeze_backbone:
        param_groups.append(
            {"params": model.backbone.parameters(), "lr": args.lr_backbone}
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    best_acc = -1.0
    best_path = ckpt_dir / "best_finetuned_dinov2.pth"
    mapping_path = ckpt_dir / "class_to_idx.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        logger.info(
            "Epoch %d/%d | train_loss %.4f train_acc %.2f%% | val_loss %.4f val_acc %.2f%%",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "best_val_acc": best_acc,
                },
                best_path,
            )
            logger.info("Saved new best checkpoint: %s (acc=%.2f%%)", best_path, best_acc)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    logger.info("Training complete. Best validation accuracy: %.2f%%", best_acc)
    logger.info("Best checkpoint: %s", best_path)
    logger.info("Class mapping: %s", mapping_path)


if __name__ == "__main__":
    main()
