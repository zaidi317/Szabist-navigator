# Campus Navigation App - Backend

A sophisticated campus navigation system using DINOv2 image embeddings and FAISS-based location identification with A* pathfinding.

## Features

- 🏫 **Location Identification**: Uses pre-trained DINOv2 ViT-S/14 model to identify campus locations from images
- 🔍 **FAISS-based Search**: Efficient similarity search using FAISS IndexFlatL2
- 📍 **Top-5 Voting**: Robust location identification using voting mechanism
- 🛤️ **A* Pathfinding**: Optimal route calculation between campus locations
- ⏱️ **Travel Time Estimation**: Estimates walking time based on path distance
- 🗺️ **Campus Graph**: Pre-configured graph of campus locations and connections

## Project Structure

```
campus3/
├── app.py                 # Main application with inference functions
├── config.py              # Configuration and constants
├── embedding_model.py     # DINOv2 embedder
├── faiss_index.py         # FAISS indexing and searching
├── pathfinding.py         # A* algorithm implementation
├── data_loader.py         # Dataset loading utilities
├── train_index.py         # Script to build FAISS index
├── requirements.txt       # Python dependencies
├── dataset/
│   ├── train/
│   │   ├── Cafe/
│   │   ├── Courtyard/
│   │   ├── FountainArea/
│   │   ├── helmetarea/
│   │   ├── SSC/
│   │   └── stairs/
│   └── test/
│       ├── Cafe/
│       ├── Courtyard/
│       ├── FountainArea/
│       ├── helmetarea/
│       ├── SSC/
│       └── stairs/
├── campus.index           # FAISS index (generated)
├── labels.npy             # Location labels (generated)
└── metadata.json          # Index metadata (generated)
```

## Installation

### 1. Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download DINOv2 Model

The DINOv2 model will be automatically downloaded on first run via `torch.hub`.

## Quick Start

### Step 1: Build the FAISS Index

Before using the app, you must build the FAISS index from your training dataset:

```bash
python train_index.py
```

This script will:
- Load all images from `dataset/train/`
- Generate DINOv2 embeddings for each image
- Create a FAISS index (`campus.index`)
- Save location labels (`labels.npy`)
- Save metadata (`metadata.json`)

**Expected output:**
```
[INFO] Starting FAISS Index Building Process
[INFO] Initializing DINOv2 embedder...
[INFO] Loading dataset...
[INFO] Found 300 images
[INFO] Generating embeddings...
[INFO] Building FAISS index...
[INFO] Index building completed successfully!
```

### Step 2: Use the App

```python
from app import CampusNavigationApp

# Initialize the app
app = CampusNavigationApp()

# Example 1: Identify location from an image
result = app.identify_location("path/to/image.jpg")
print(f"Location: {result['location']}")
print(f"Confidence: {result['confidence']}")
print(f"Top 5 matches: {result['top_5_matches']}")

# Example 2: Get navigation path
path_result = app.get_navigation_path("Cafe", "Library")
print(f"Path: {' -> '.join(path_result['path'])}")
print(f"Distance: {path_result['total_distance_meters']}m")
print(f"Time: {path_result['estimated_time_minutes']:.1f} minutes")

# Example 3: Full navigation query (identify location + get path)
full_result = app.full_navigation_query("path/to/image.jpg", "Cafe")
location_result = full_result['location_identification']
navigation_result = full_result['navigation']
```

## Core Modules

### 1. `config.py`
Configuration constants including:
- Dataset paths
- Model parameters
- Index parameters
- Campus locations

### 2. `embedding_model.py`
**Class: `DINOv2Embedder`**

Handles image embedding generation using DINOv2:
- `get_embedding(image_path)` - Generate embedding for single image
- `get_embeddings_batch(image_paths, batch_size)` - Generate embeddings for multiple images

### 3. `faiss_index.py`
**Class: `FAISSIndex`**

Manages FAISS indexing:
- `add_embeddings(embeddings, labels)` - Add embeddings to index
- `search(query_embedding, k)` - Find k nearest neighbors
- `save()` / `load()` - Persist/load index

### 4. `pathfinding.py`
**Classes: `CampusGraph`, `AStarPathfinder`**

Implements A* pathfinding:
- `find_path(start, end)` - Find optimal path using A*
- `estimate_travel_time(path)` - Calculate walking time
- Customizable campus graph with coordinates

### 5. `data_loader.py`
**Class: `DataLoader`**

Dataset utilities:
- `get_images_by_location()` - Load images organized by location
- `get_all_locations()` - Get list of available locations
- `count_images()` - Get image count per location

### 6. `app.py`
**Class: `CampusNavigationApp`**

Main application class with:
- `identify_location(image_path)` - Identify location from image (with top-5 voting)
- `get_navigation_path(start, end)` - Get path between locations
- `full_navigation_query(image_path, destination)` - Complete navigation query
- `get_index_statistics()` - Get index information

## API Usage Examples

### Identify Location with Confidence

```python
from app import CampusNavigationApp

app = CampusNavigationApp()

result = app.identify_location("test_image.jpg")

# Returns:
{
    "location": "Cafe",
    "confidence": 0.8,
    "top_5_matches": ["Cafe", "Courtyard", "Cafe", "Cafe", "SSC"],
    "distances": [150.5, 234.2, 245.1, 301.3, 456.7],
    "voting_results": {"Cafe": 3, "Courtyard": 1, "SSC": 1}
}
```

### Get Navigation Path

```python
result = app.get_navigation_path("Cafe", "Library")

# Returns:
{
    "start": "Cafe",
    "end": "Library",
    "path": ["Cafe", "Courtyard", "SSC", "Library"],
    "num_locations": 4,
    "total_distance_meters": 450,
    "estimated_time_seconds": 300,
    "estimated_time_minutes": 5.0
}
```

### Full Navigation Query

```python
result = app.full_navigation_query("current_location.jpg", "Destination")

# Returns comprehensive result:
{
    "location_identification": {
        "location": "Cafe",
        "confidence": 0.9,
        ...
    },
    "navigation": {
        "start": "Cafe",
        "end": "Destination",
        "path": [...],
        ...
    }
}
```

## Configuration Options

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "dinov2_vits14"
DEVICE = "cuda"  # or "cpu"

# Search
TOP_K = 5  # Number of neighbors for voting
DISTANCE_THRESHOLD = 1000  # Max L2 distance

# Processing
BATCH_SIZE = 8
IMAGE_SIZE = 224

# Pathfinding
PATHFINDING_HEURISTIC_WEIGHT = 1.0
```

## Customizing the Campus Graph

Edit the `CampusGraph` class in `pathfinding.py`:

```python
# Add new location
graph.add_edge("Cafe", "NewLocation", distance=200)

# Update coordinates for better heuristic
graph.set_coordinates("Cafe", x=0, y=0)
graph.set_coordinates("NewLocation", x=100, y=150)
```

## Performance Considerations

- **Model Size**: DINOv2 ViT-S/14 ~80MB
- **Embedding Generation**: ~10-20ms per image on GPU
- **FAISS Search**: <1ms for top-5 search on 300 images
- **A* Pathfinding**: <1ms for typical campus graphs

## Troubleshooting

### Missing FAISS Index
**Error**: `FileNotFoundError: FAISS index files not found`

**Solution**: Run `python train_index.py` to build the index

### GPU Out of Memory
**Solution**: 
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU: Set `DEVICE = "cpu"` in `config.py`

### Slow Image Processing
**Solution**:
- Use GPU if available
- Increase `BATCH_SIZE` (if GPU memory allows)

## Testing

Run the demo:

```bash
python app.py
```

This will:
1. Initialize the app
2. Display available locations
3. Show index statistics
4. Demonstrate location identification
5. Show navigation path example

## Advanced Usage

### Batch Processing Multiple Images

```python
from pathlib import Path
from app import CampusNavigationApp

app = CampusNavigationApp()

image_dir = Path("test_images")
for image_path in image_dir.glob("*.jpg"):
    result = app.identify_location(str(image_path))
    print(f"{image_path.name}: {result['location']} ({result['confidence']:.2f})")
```

### Custom Campus Graph

```python
from pathfinding import CampusGraph, AStarPathfinder

# Create custom graph
graph = CampusGraph()

# Add custom connections
graph.add_edge("Cafe", "Lounge", distance=100)
graph.add_edge("Lounge", "Library", distance=150)

# Use pathfinder
pathfinder = AStarPathfinder(graph)
path = pathfinder.find_path("Cafe", "Library")
```

## Technical Details

### DINOv2 Model
- **Architecture**: Vision Transformer (ViT-S/14)
- **Input**: RGB images (224x224 recommended)
- **Output**: 384-dimensional embeddings
- **Pre-training**: Trained on large-scale unlabeled image data
- **Reference**: [DINOv2 Paper](https://arxiv.org/abs/2304.07193)

### FAISS Index
- **Type**: IndexFlatL2 (Flat index with L2 distance)
- **Distance Metric**: Euclidean (L2) distance
- **Search**: Brute-force nearest neighbor search

### A* Pathfinding
- **Heuristic**: Euclidean distance between locations
- **Optimality**: Guaranteed to find optimal path
- **Complexity**: O(n log n) where n = number of locations

## Requirements

- `torch` ≥ 2.0.0
- `torchvision` ≥ 0.15.0
- `faiss-cpu` ≥ 1.7.4
- `numpy` ≥ 1.21.0
- `Pillow` ≥ 9.0.0
- `tqdm` ≥ 4.60.0

## Future Enhancements

- [ ] Real-time GPS integration
- [ ] Multiple routing strategies (fastest, scenic, least crowded)
- [ ] Indoor positioning with WiFi signals
- [ ] Mobile app integration
- [ ] Multi-floor support
- [ ] Crowding information
- [ ] Accessibility routing
- [ ] Building layout visualization

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the example code in `app.py`
3. Check logs with `logging.basicConfig(level=logging.DEBUG)`

---

**Last Updated**: 2024
**Python Version**: 3.8+
**Status**: Production Ready
