"""
Training script: Build FAISS index from campus images
Run this script to create the index before using the app
"""

import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

# Import modules
from config import (
    TRAIN_DIR, INDEX_FILE, LABELS_FILE, METADATA_FILE,
    MODEL_NAME, DEVICE, BATCH_SIZE
)
from embedding_model import DINOv2Embedder
from faiss_index import FAISSIndex
from data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index():
    """
    Build FAISS index from training dataset
    """
    logger.info("=" * 60)
    logger.info("Starting FAISS Index Building Process")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("\n[1/4] Initializing DINOv2 embedder...")
    embedder = DINOv2Embedder(model_name=MODEL_NAME, device=DEVICE)
    
    logger.info("\n[2/4] Loading dataset...")
    data_loader = DataLoader(TRAIN_DIR.parent)
    image_paths, labels = data_loader.get_images_by_location(split="train")
    
    if not image_paths:
        logger.error("No images found in dataset!")
        return False
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Count images per location
    logger.info("\nImages per location:")
    image_dict = data_loader.get_images_by_location_dict(split="train")
    for loc, images in sorted(image_dict.items()):
        logger.info(f"  {loc}: {len(images)} images")
    
    # Generate embeddings
    logger.info("\n[3/4] Generating embeddings...")
    embeddings = embedder.get_embeddings_batch(image_paths, batch_size=BATCH_SIZE)
    
    if not embeddings:
        logger.error("Failed to generate embeddings!")
        return False
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding shape: {embeddings_array.shape}")
    
    # Build FAISS index
    logger.info("\n[4/4] Building FAISS index...")
    faiss_index = FAISSIndex(embedding_dim=embeddings_array.shape[1])
    
    # Add embeddings in batches to track progress
    for i in tqdm(range(0, len(embeddings_array), BATCH_SIZE), desc="Adding embeddings"):
        batch_end = min(i + BATCH_SIZE, len(embeddings_array))
        batch_embeddings = embeddings_array[i:batch_end]
        batch_labels = labels[i:batch_end]
        
        faiss_index.add_embeddings(batch_embeddings, batch_labels)
    
    # Store metadata
    for path, label in zip(image_paths, labels):
        faiss_index.metadata["image_path"].append(str(path))
        faiss_index.metadata["location"].append(label)
    
    # Save index
    logger.info("\nSaving index and labels...")
    faiss_index.save(INDEX_FILE, LABELS_FILE, METADATA_FILE)
    
    logger.info("\n" + "=" * 60)
    logger.info("Index building completed successfully!")
    logger.info(f"Index saved to: {INDEX_FILE}")
    logger.info(f"Labels saved to: {LABELS_FILE}")
    logger.info(f"Metadata saved to: {METADATA_FILE}")
    logger.info(f"Total items indexed: {faiss_index.get_total_items()}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = build_index()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error during index building: {e}", exc_info=True)
        sys.exit(1)
