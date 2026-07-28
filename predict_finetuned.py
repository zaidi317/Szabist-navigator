"""
Inference with fine-tuned DINOv2 checkpoint.

Usage:
python predict_finetuned.py --image dataset/test/Cafe/xxx.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

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
        return self.classifier(features)


class FineTunedLocationPredictor:
    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        self.model_name = ckpt["model_name"]
        self.class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
        self.idx_to_class: Dict[int, str] = {int(k): v for k, v in ckpt["idx_to_class"].items()} \
            if any(isinstance(k, str) for k in ckpt["idx_to_class"].keys()) else ckpt["idx_to_class"]

        num_classes = ckpt["num_classes"]

        self.model = DINOv2Classifier(num_classes=num_classes, model_name=self.model_name)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, image_path: str | Path, top_k: int = 5) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        top_k = min(top_k, probs.shape[1])
        values, indices = torch.topk(probs, k=top_k, dim=1)

        predictions = []
        for p, idx in zip(values[0].tolist(), indices[0].tolist()):
            label = self.idx_to_class[int(idx)]
            predictions.append({"label": label, "probability": float(p)})

        return {
            "image_path": str(image_path),
            "predicted_location": predictions[0]["label"],
            "confidence": predictions[0]["probability"],
            "top_predictions": predictions,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with fine-tuned DINOv2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_finetuned_dinov2.pth",
        help="Fine-tuned checkpoint path",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    predictor = FineTunedLocationPredictor(args.checkpoint)
    out = predictor.predict(args.image, top_k=args.top_k)
    logger.info("Prediction: %s", out)


if __name__ == "__main__":
    main()
