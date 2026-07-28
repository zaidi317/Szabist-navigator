# Campus Navigation App - API Documentation

## Overview

The Campus Navigation App backend provides a comprehensive Python API for location identification and pathfinding. This document details all available functions and classes.

## Main Classes

### CampusNavigationApp

Main application class that coordinates all functionality.

#### Initialization

```python
from app import CampusNavigationApp

# Initialize the app (loads model, index, pathfinder)
app = CampusNavigationApp()

# Optional: specify custom index directory
app = CampusNavigationApp(index_dir=Path("custom/path"))
```

#### Methods

---

### `identify_location(image_path: str, use_voting: bool = True) -> Dict`

Identifies the campus location from an image.

**Parameters:**
- `image_path` (str): Path to the query image
- `use_voting` (bool, default=True): Use top-5 voting mechanism or single best match

**Returns:** Dictionary containing:
- `location` (str): Identified location name
- `confidence` (float): Confidence score 0-1
- `top_5_matches` (list): Top 5 nearest locations
- `distances` (list): Distances to top 5 matches
- `voting_results` (dict): Vote count per location (if use_voting=True)
- `error` (str): Error message if any

**Example:**
```python
result = app.identify_location("test_image.jpg")
print(f"Location: {result['location']}")
print(f"Confidence: {result['confidence']:.1%}")
```

**Output:**
```python
{
    'image_path': '/path/to/image.jpg',
    'location': 'Cafe',
    'confidence': 0.8,
    'top_5_matches': ['Cafe', 'Courtyard', 'Cafe', 'Cafe', 'SSC'],
    'distances': [150.5, 234.2, 245.1, 301.3, 456.7],
    'voting_results': {'Cafe': 3, 'Courtyard': 1, 'SSC': 1}
}
```

---

### `get_navigation_path(start_location: str, end_location: str) -> Dict`

Finds optimal path between two campus locations using A* algorithm.

**Parameters:**
- `start_location` (str): Starting location name
- `end_location` (str): Destination location name

**Returns:** Dictionary containing:
- `start` (str): Starting location
- `end` (str): Destination location
- `path` (list): Ordered list of locations from start to end
- `num_locations` (int): Number of locations in path
- `total_distance_meters` (float): Total distance in meters
- `estimated_time_seconds` (float): Estimated walk time in seconds
- `estimated_time_minutes` (float): Estimated walk time in minutes
- `error` (str): Error message if path not found

**Example:**
```python
result = app.get_navigation_path("Cafe", "Library")

if 'error' not in result:
    print(f"Route: {' → '.join(result['path'])}")
    print(f"Distance: {result['total_distance_meters']}m")
    print(f"Time: {result['estimated_time_minutes']:.1f} min")
```

**Output:**
```python
{
    'start': 'Cafe',
    'end': 'Library',
    'path': ['Cafe', 'Courtyard', 'SSC', 'Library'],
    'num_locations': 4,
    'total_distance_meters': 450,
    'estimated_time_seconds': 300.0,
    'estimated_time_minutes': 5.0
}
```

---

### `full_navigation_query(image_path: str, destination: str) -> Dict`

Complete navigation workflow: identify current location and get path to destination.

**Parameters:**
- `image_path` (str): Path to image showing current location
- `destination` (str): Target location name

**Returns:** Dictionary containing two keys:
- `location_identification` (dict): Result from identify_location()
- `navigation` (dict): Result from get_navigation_path()

**Example:**
```python
result = app.full_navigation_query("my_location.jpg", "Library")

current = result['location_identification']
nav = result['navigation']

print(f"You are at: {current['location']}")
print(f"Route to {nav['end']}: {' → '.join(nav['path'])}")
```

**Output:**
```python
{
    'location_identification': {
        'location': 'Cafe',
        'confidence': 0.9,
        'top_5_matches': ['Cafe', 'Cafe', 'Courtyard', 'Cafe', 'SSC'],
        'distances': [145.2, 152.1, 234.5, 298.3, 412.1],
        'voting_results': {'Cafe': 3, 'Courtyard': 1, 'SSC': 1}
    },
    'navigation': {
        'start': 'Cafe',
        'end': 'Library',
        'path': ['Cafe', 'Courtyard', 'SSC', 'Library'],
        'total_distance_meters': 450,
        'estimated_time_minutes': 5.0
    }
}
```

---

### `get_available_locations() -> List[str]`

Get list of all available campus locations.

**Returns:** List of location names

**Example:**
```python
locations = app.get_available_locations()
# Returns: ['Cafe', 'Courtyard', 'FountainArea', 'helmetarea', 'SSC', 'stairs']
```

---

### `get_index_statistics() -> Dict`

Get statistics about the FAISS index.

**Returns:** Dictionary containing:
- `total_indexed_items` (int): Total images in index
- `embedding_dimension` (int): Dimension of embeddings
- `locations` (list): All location names
- `items_per_location` (dict): Image count per location
- `index_file` (str): Path to index file
- `labels_file` (str): Path to labels file

**Example:**
```python
stats = app.get_index_statistics()
print(f"Total images: {stats['total_indexed_items']}")
for loc, count in stats['items_per_location'].items():
    print(f"{loc}: {count} images")
```

---

## Utility Classes

### DINOv2Embedder

Generates image embeddings using the DINOv2 model.

**Location:** `embedding_model.py`

#### Methods

##### `get_embedding(image_path: Union[str, Path]) -> np.ndarray`

Generate embedding for a single image.

**Parameters:**
- `image_path`: Path to image file

**Returns:** 1D numpy array of shape (384,)

**Example:**
```python
from embedding_model import DINOv2Embedder

embedder = DINOv2Embedder()
embedding = embedder.get_embedding("image.jpg")
print(embedding.shape)  # (384,)
```

---

##### `get_embeddings_batch(image_paths: List[Union[str, Path]], batch_size: int = 8) -> List[np.ndarray]`

Generate embeddings for multiple images efficiently.

**Parameters:**
- `image_paths`: List of image file paths
- `batch_size`: Batch size for processing

**Returns:** List of embeddings as numpy arrays

**Example:**
```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
embeddings = embedder.get_embeddings_batch(images, batch_size=4)
```

---

### FAISSIndex

Manages FAISS indexing and searching.

**Location:** `faiss_index.py`

#### Methods

##### `add_embeddings(embeddings: np.ndarray, labels: List[str])`

Add embeddings to index.

**Parameters:**
- `embeddings`: Array of shape (N, 384)
- `labels`: Location names corresponding to embeddings

---

##### `search(query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[str]]`

Search for nearest neighbors.

**Parameters:**
- `query_embedding`: Query embedding (1D array)
- `k`: Number of neighbors to return

**Returns:** Tuple of (distances, labels)

---

##### `save(index_path: Path, labels_path: Path, metadata_path: Path = None)`

Save index to disk.

---

##### `load(index_path: Path, labels_path: Path, metadata_path: Path = None) -> FAISSIndex` (static)

Load index from disk.

---

### CampusGraph

Represents campus locations and connections.

**Location:** `pathfinding.py`

#### Properties

- `graph`: Dict mapping location to neighbors with distances
- `coordinates`: Dict mapping location to (x, y) coordinates

#### Methods

##### `add_edge(from_loc: str, to_loc: str, distance: float)`

Add bidirectional connection between locations.

**Example:**
```python
from pathfinding import CampusGraph

graph = CampusGraph()
graph.add_edge("Cafe", "Library", distance=200)
```

---

##### `set_coordinates(location: str, x: float, y: float)`

Set coordinates for a location (used for pathfinding heuristic).

---

##### `get_neighbors(location: str) -> List[Tuple[str, float]]`

Get neighboring locations with distances.

---

### AStarPathfinder

A* pathfinding algorithm implementation.

**Location:** `pathfinding.py`

#### Methods

##### `find_path(start: str, end: str, heuristic_weight: float = 1.0) -> Optional[List[str]]`

Find optimal path using A* algorithm.

**Parameters:**
- `start`: Starting location
- `end`: Destination location
- `heuristic_weight`: Heuristic weight (higher = more aggressive)

**Returns:** List of locations representing the path, or None if no path exists

**Example:**
```python
from pathfinding import CampusGraph, AStarPathfinder

graph = CampusGraph()
pathfinder = AStarPathfinder(graph)
path = pathfinder.find_path("Cafe", "Library")
```

---

##### `estimate_travel_time(path: List[str], speed_mps: float = 1.5) -> float`

Estimate walking time for a path.

**Parameters:**
- `path`: List of locations
- `speed_mps`: Walking speed in m/s (default: 1.5)

**Returns:** Time in seconds

---

### DataLoader

Loads images from dataset.

**Location:** `data_loader.py`

#### Methods

##### `get_images_by_location(split: str = "train") -> Tuple[List[Path], List[str]]`

Get all images with their location labels.

**Returns:** Tuple of (image_paths, labels)

---

##### `get_images_by_location_dict(split: str = "train") -> dict`

Get images organized as {location: [paths]}.

---

##### `get_all_locations(split: str = "train") -> List[str]`

Get all location names in dataset.

---

##### `count_images(split: str = "train") -> dict`

Get image count per location.

---

## Advanced Usage

### Custom Voting Mechanism

```python
from collections import Counter
from embedding_model import DINOv2Embedder
from faiss_index import FAISSIndex

embedder = DINOv2Embedder()
index = FAISSIndex.load("campus.index", "labels.npy")

# Get embedding
query_embedding = embedder.get_embedding("image.jpg")

# Search (top-10 instead of top-5)
distances, neighbors = index.search(query_embedding, k=10)

# Custom voting (top-7 with weighting)
votes = {}
for i, neighbor in enumerate(neighbors[:7]):
    weight = 1.0 / (i + 1)  # Closer matches have higher weight
    votes[neighbor] = votes.get(neighbor, 0) + weight

identified = max(votes, key=votes.get)
```

### Batch Processing

```python
from pathlib import Path

app = CampusNavigationApp()

image_dir = Path("test_images")
results = []

for image_path in image_dir.glob("*.jpg"):
    result = app.identify_location(str(image_path))
    results.append({
        'image': image_path.name,
        'location': result['location'],
        'confidence': result['confidence']
    })

# Filter by confidence
high_confidence = [r for r in results if r['confidence'] > 0.7]
```

### Custom Campus Graph

```python
from pathfinding import CampusGraph, AStarPathfinder

# Create empty graph
graph = CampusGraph()
graph.graph = {}  # Clear default graph

# Add custom locations
locations = {
    "entrance": (0, 0),
    "library": (100, 50),
    "cafeteria": (50, -100),
}

for loc, (x, y) in locations.items():
    graph.coordinates[loc] = (x, y)
    graph.graph[loc] = []

# Add connections
edges = [
    ("entrance", "library", 120),
    ("entrance", "cafeteria", 150),
    ("library", "cafeteria", 180),
]

for from_loc, to_loc, dist in edges:
    graph.add_edge(from_loc, to_loc, dist)

# Use pathfinder
pathfinder = AStarPathfinder(graph)
path = pathfinder.find_path("entrance", "library")
```

---

## Error Handling

### Catching Errors

```python
try:
    result = app.identify_location("missing_file.jpg")
    if 'error' in result:
        print(f"Error: {result['error']}")
except Exception as e:
    print(f"Exception: {e}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: FAISS index files not found` | Index not built | Run `python train_index.py` |
| `FileNotFoundError: Image not found` | Invalid image path | Check image path exists |
| `No path found from X to Y` | Disconnected graph | Add edge between locations |
| `Failed to generate embedding` | Invalid image format | Use JPG, PNG, or similar |

---

## Performance Tips

1. **Batch Processing**: Use `get_embeddings_batch()` for multiple images
2. **GPU Acceleration**: Ensure CUDA is available for faster inference
3. **Caching**: Cache embeddings if querying same images repeatedly
4. **Batch Size**: Increase `BATCH_SIZE` in config (if GPU memory allows)

---

## Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "dinov2_vits14"
DEVICE = "cuda"  # or "cpu"

# Search
TOP_K = 5
DISTANCE_THRESHOLD = 1000

# Processing
BATCH_SIZE = 8
IMAGE_SIZE = 224

# Pathfinding
PATHFINDING_HEURISTIC_WEIGHT = 1.0
```

---

## Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = CampusNavigationApp()  # Now shows detailed logs
```

---

## Thread Safety

Current implementation is NOT thread-safe for concurrent inference. For multi-threaded applications:

```python
from threading import Lock

class ThreadSafeApp:
    def __init__(self):
        self.app = CampusNavigationApp()
        self.lock = Lock()
    
    def identify_location(self, image_path):
        with self.lock:
            return self.app.identify_location(image_path)
```

---

## Type Hints

All functions use type hints for better IDE support:

```python
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def identify_location(image_path: str) -> Dict:
    ...

def get_navigation_path(start: str, end: str) -> Dict:
    ...
```

---

## References

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [A* Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)

