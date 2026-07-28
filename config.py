"""
Configuration and constants for Campus Navigation app
"""

from pathlib import Path

# Dataset paths (Windows-compatible)
BASE_DIR = Path(__file__).parent.absolute()
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Index and labels paths
INDEX_FILE = BASE_DIR / "campus.index"
LABELS_FILE = BASE_DIR / "labels.npy"
METADATA_FILE = BASE_DIR / "metadata.json"

# Model configuration
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"  # ViT-B/14 fine-tuned via timm
EMBEDDING_DIM = 768  # Output dimension of ViT-B/14
CHECKPOINT = BASE_DIR / "artifacts" / "dinov2_finetuned_best.pth"
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Search configuration
TOP_K = 5  # Number of nearest neighbors for voting mechanism
DISTANCE_THRESHOLD = 1000  # Maximum L2 distance for valid matches

# Campus locations (auto-discovered from dataset)
CAMPUS_LOCATIONS = [
    "Cafe",
    "Courtyard",
    "FountainArea",
    "helmetarea",
    "SSC",
    "stairs"
]

# Image extensions to process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Image preprocessing
IMAGE_SIZE = 224  # DINOv2 expects 224x224 input

# Batch size for embedding generation
BATCH_SIZE = 8

# A* pathfinding configuration
PATHFINDING_HEURISTIC_WEIGHT = 1.0  # Weight for heuristic in A*

# API
API_VERSION = "1.1.0"
CORS_ORIGINS = "*"  # Set to a specific origin in production
