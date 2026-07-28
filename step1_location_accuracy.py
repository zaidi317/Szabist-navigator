"""
Step 1 backend for Campus Navigation:
1) Identify location from a single image
2) Evaluate accuracy on dataset/test
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from config import DEVICE, INDEX_FILE, LABELS_FILE, MODEL_NAME, TOP_K
from embedding_model import DINOv2Embedder
from faiss_index import FAISSIndex


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class LocationIdentifier:
    """Image location predictor using DINOv2 embeddings + FAISS top-k voting."""

    def __init__(
        self,
        index_path: Path = INDEX_FILE,
        labels_path: Path = LABELS_FILE,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        top_k: int = TOP_K,
    ) -> None:
        self.top_k = max(1, top_k)

        if not index_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "Index files not found. Run 'python train_index.py' first. "
                f"Missing: {index_path} or {labels_path}"
            )

        logger.info("Loading DINOv2 model...")
        self.embedder = DINOv2Embedder(model_name=model_name, device=device)

        logger.info("Loading FAISS index and labels...")
        self.index = FAISSIndex.load(index_path=index_path, labels_path=labels_path)

    def identify_location(self, image_path: str | Path) -> Dict:
        """Identify location from one image using top-k voting."""
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        embedding = self.embedder.get_embedding(image_path)
        if embedding is None:
            return {"error": f"Failed to create embedding for: {image_path}"}

        distances, neighbor_labels = self.index.search(embedding, k=self.top_k)

        votes = Counter(neighbor_labels)
        predicted_label, vote_count = votes.most_common(1)[0]
        confidence = vote_count / float(self.top_k)

        return {
            "image_path": str(image_path),
            "predicted_location": predicted_label,
            "confidence": round(confidence, 4),
            "top_matches": neighbor_labels,
            "distances": [float(d) for d in distances.tolist()],
            "votes": dict(votes),
        }

    def evaluate_test_accuracy(self, test_dir: str | Path) -> Dict:
        """Evaluate overall and per-class accuracy on dataset/test structure."""
        test_dir = Path(test_dir)
        if not test_dir.exists():
            return {"error": f"Test directory not found: {test_dir}"}

        class_dirs = [d for d in sorted(test_dir.iterdir()) if d.is_dir()]
        if not class_dirs:
            return {"error": f"No class folders found in: {test_dir}"}

        total = 0
        correct = 0
        per_class_total: Dict[str, int] = defaultdict(int)
        per_class_correct: Dict[str, int] = defaultdict(int)
        mistakes: List[Tuple[str, str, str]] = []

        for class_dir in class_dirs:
            true_label = class_dir.name
            image_paths: List[Path] = []
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(class_dir.glob(f"*{ext}"))
                image_paths.extend(class_dir.glob(f"*{ext.upper()}"))

            for image_path in sorted(image_paths):
                result = self.identify_location(image_path)
                if "error" in result:
                    continue

                pred = result["predicted_location"]
                total += 1
                per_class_total[true_label] += 1

                if pred == true_label:
                    correct += 1
                    per_class_correct[true_label] += 1
                else:
                    mistakes.append((str(image_path), true_label, pred))

        if total == 0:
            return {"error": "No test images found to evaluate."}

        per_class_accuracy = {
            label: round(
                (per_class_correct[label] / per_class_total[label]) * 100.0, 2
            )
            for label in sorted(per_class_total.keys())
            if per_class_total[label] > 0
        }

        overall_accuracy = round((correct / total) * 100.0, 2)

        return {
            "test_dir": str(test_dir),
            "top_k": self.top_k,
            "total_images": total,
            "correct_predictions": correct,
            "overall_accuracy_percent": overall_accuracy,
            "per_class_accuracy_percent": per_class_accuracy,
            "num_mistakes": len(mistakes),
            "sample_mistakes": mistakes[:20],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: location prediction + test accuracy evaluation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for location identification",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Path to test directory (default: dataset/test)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip test-set accuracy evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-k neighbors for voting (default: {TOP_K})",
    )

    args = parser.parse_args()

    model = LocationIdentifier(top_k=args.top_k)

    if args.image:
        logger.info("Running single-image location identification...")
        result = model.identify_location(args.image)
        logger.info("Prediction result: %s", result)

    if not args.no_eval:
        logger.info("Running test-set accuracy evaluation...")
        metrics = model.evaluate_test_accuracy(args.test_dir)
        logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
