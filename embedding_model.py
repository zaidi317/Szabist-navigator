"""
DINOv2 embedding model — fine-tuned ViT-B/14 loaded via timm.
Inference: backbone embed → L2-normalise → FAISS cosine retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class DINOv2Embedder:
    """
    Fine-tuned DINOv2 ViT-B/14 backbone loaded from a timm checkpoint.
    Produces L2-normalised 768-d embeddings for FAISS cosine retrieval.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading DINOv2Embedder from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg        = ckpt.get("cfg", {})
        model_name = cfg.get("model_name", "vit_base_patch14_dinov2.lvd142m")
        self.classes    = ckpt["classes"]
        self.num_classes = len(self.classes)

        # img_size=224 matches training; timm defaults this model to 518
        img_size = cfg.get("img_size", 224)
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size
        )
        self.embed_dim = self.backbone.num_features  # 768 for ViT-B/14

        # Load backbone weights
        backbone_sd = {k[len("backbone."):]: v
                       for k, v in ckpt["model_state"].items()
                       if k.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_sd)
        self.backbone.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        logger.info(
            f"DINOv2Embedder ready — {self.num_classes} classes, "
            f"embed_dim={self.embed_dim}, "
            f"val_acc={ckpt.get('val_acc', '?'):.4f}"
        )

    # ── core method ───────────────────────────────────────────────────────────

    def embed(self, pil_image: Image.Image) -> np.ndarray:
        """Return (1, 768) L2-normalised embedding as float32 numpy."""
        x = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(x)
            feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().astype(np.float32)

    # ── compatibility aliases ─────────────────────────────────────────────────

    def get_embedding_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """1-D (768,) alias used by the FAISS search path."""
        return self.embed(pil_image)[0]

    def get_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from disk and return 1-D (768,) embedding."""
        try:
            img = Image.open(image_path).convert("RGB")
            return self.embed(img)[0]
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            return None

    def get_embeddings_batch(self, image_paths: List[Union[str, Path]],
                             batch_size: int = 8) -> List[np.ndarray]:
        """Batch embedding — returns list of 1-D (768,) arrays."""
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_imgs, valid = [], []
            for p in image_paths[i:i + batch_size]:
                try:
                    batch_imgs.append(self.transform(Image.open(p).convert("RGB")))
                    valid.append(p)
                except Exception:
                    pass
            if not batch_imgs:
                continue
            batch_t = torch.stack(batch_imgs).to(self.device)
            with torch.no_grad():
                feats = self.backbone(batch_t)
                feats = F.normalize(feats, p=2, dim=1)
            embeddings.extend(feats.cpu().numpy().astype(np.float32))
            logger.info(f"Batch {i // batch_size + 1}: {len(valid)} images")
        return embeddings
