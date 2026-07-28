# Campus Navigation App - Getting Started Guide

## 📦 What You Have

A complete, production-ready Campus Navigation backend with:

- ✅ DINOv2 image embeddings for location identification
- ✅ FAISS-based semantic search
- ✅ A* pathfinding algorithm
- ✅ Modular, well-documented code
- ✅ REST API with Flask
- ✅ Comprehensive examples

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This requires PyTorch and FAISS. First-time installation might take a few minutes.

### 2. Verify Setup

```bash
python setup_check.py
```

This checks:
- ✓ All packages installed
- ✓ Dataset directory structure
- ✓ Python modules loadable
- ✓ FAISS index exists

### 3. Build FAISS Index

```bash
python train_index.py
```

This:
- Loads all ~300 images from `dataset/train/`
- Generates DINOv2 embeddings
- Creates `campus.index` (FAISS index)
- Saves `labels.npy` (location labels)

⏱️ **Time:** 5-15 minutes depending on GPU availability

### 4. Test the App

```bash
python examples.py
```

This demonstrates all key features with your actual dataset.

## 📁 Project Structure

```
campus3/
├── app.py                 # ⭐ Main application (START HERE)
├── train_index.py         # Build FAISS index from dataset
├── flask_api.py          # REST API server
├── examples.py           # Usage examples
├── setup_check.py        # Verify installation
│
├── config.py             # Configuration constants
├── embedding_model.py    # DINOv2 embeddings
├── faiss_index.py        # FAISS indexing
├── pathfinding.py        # A* pathfinding
├── data_loader.py        # Dataset utilities
│
├── dataset/
│   ├── train/           # Training images (50 per location)
│   └── test/            # Test images
│
├── campus.index          # Generated FAISS index
├── labels.npy            # Generated labels
└── metadata.json         # Generated metadata
```

## 🎯 Core Functionality

### 1. Location Identification

```python
from app import CampusNavigationApp

app = CampusNavigationApp()
result = app.identify_location("photo.jpg")

print(result['location'])        # "Cafe"
print(result['confidence'])      # 0.8
print(result['top_5_matches'])   # ['Cafe', 'Cafe', ...]
```

### 2. Path Finding

```python
result = app.get_navigation_path("Cafe", "Library")

print(result['path'])                    # ['Cafe', 'Courtyard', 'Library']
print(result['total_distance_meters'])   # 450
print(result['estimated_time_minutes'])  # 5.0
```

### 3. Complete Navigation

```python
result = app.full_navigation_query("my_photo.jpg", "Library")

# Returns both location identification AND path
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
MODEL_NAME = "dinov2_vits14"           # Model to use
DEVICE = "cuda"                        # "cuda" or "cpu"
TOP_K = 5                              # Voting neighbors
BATCH_SIZE = 8                         # Processing batch size
```

## 🌐 REST API

Start the server:

```bash
python flask_api.py
```

Then use from frontend:

```javascript
// Get locations
GET http://localhost:5000/api/locations

// Identify location
POST http://localhost:5000/api/identify
  file: image.jpg

// Get path
POST http://localhost:5000/api/navigate
  {"start": "Cafe", "end": "Library"}

// Full navigation
POST http://localhost:5000/api/full-navigate
  file: image.jpg
  destination: "Library"
```

## 📚 Documentation

- **[README.md](README.md)** - Comprehensive documentation
- **[API_DOCS.md](API_DOCS.md)** - Detailed API reference
- **[examples.py](examples.py)** - Runnable usage examples

## 🎓 How It Works

### Architecture

```
┌─────────────────────────────────────┐
│     Campus Navigation App           │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐   │
│  │  1. Image Input              │   │
│  │     ↓                        │   │
│  │  2. DINOv2 Embedding        │   │
│  │     ↓                        │   │
│  │  3. FAISS Search            │   │
│  │     ↓                        │   │
│  │  4. Top-5 Voting            │   │
│  │     ↓                        │   │
│  │  5. Location Identified     │   │
│  │                              │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  A* Pathfinding              │   │
│  │  - Start Location            │   │
│  │  - Campus Graph              │   │
│  │  - Destination               │   │
│  │  - Optimal Route             │   │
│  │     ↓                        │   │
│  │  Return Path + Metrics       │   │
│  │  - Locations visited        │   │
│  │  - Total distance           │   │
│  │  - Est. travel time         │   │
│  │                              │   │
│  └──────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

## 📊 Performance

On typical hardware:

- **Embedding Generation:** 10-50ms per image (GPU: faster)
- **FAISS Search:** <1ms for top-5
- **Pathfinding:** <1ms for typical campus graphs
- **Full Query:** 50-200ms total

## 🛠️ Customization

### Add New Location

1. Create folder in `dataset/train/{LocationName}/`
2. Add ~50 images
3. Run `python train_index.py` to rebuild index
4. Update campus graph in `pathfinding.py`:

```python
graph.add_edge("Cafe", "NewLocation", distance=200)
graph.set_coordinates("NewLocation", x=100, y=150)
```

### Change Model

Edit `config.py`:

```python
MODEL_NAME = "dinov2_vitb14"  # Larger model
# or
MODEL_NAME = "dinov2_vitl14"  # Even larger
```

### Adjust Search Parameters

Edit `config.py`:

```python
TOP_K = 7                          # Use 7 nearest instead of 5
DISTANCE_THRESHOLD = 2000          # Increase distance tolerance
```

## ❓ Troubleshooting

### Issue: "No FAISS index found"
**Solution:** Run `python train_index.py`

### Issue: "Out of memory"
**Solution:** 
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU: `DEVICE = "cpu"`

### Issue: "Slow inference"
**Solution:**
- Use GPU if available
- Increase `BATCH_SIZE`

### Issue: "Low identification accuracy"
**Solution:**
- Add more training images (100+ per location)
- Try different model: `dinov2_vitb14`

## 📖 Learn More

```bash
# See all available examples
python examples.py

# Check API documentation
cat API_DOCS.md

# View configuration options
cat config.py

# Check implementation details
cat embedding_model.py   # DINOv2 details
cat faiss_index.py       # Search details
cat pathfinding.py       # A* details
```

## 🚀 Deployment

### Development

```bash
python app.py
```

### Production REST API

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_api:flask_app
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_api:flask_app"]
```

Build and run:

```bash
docker build -t campus-nav .
docker run -p 5000:5000 campus-nav
```

## 📝 Key Concepts

### DINOv2
- **What:** Vision Transformer pretrained on large-scale image data
- **Why:** Excellent at capturing semantic image content
- **Where:** Generates 384-dimensional embeddings per image

### FAISS
- **What:** Facebook AI Similarity Search library
- **Why:** Efficient similarity search at scale
- **How:** IndexFlatL2 uses L2 (Euclidean) distance

### A* Algorithm
- **What:** Path planning algorithm with heuristic
- **Why:** Guaranteed optimal path + efficiency
- **How:** Uses Euclidean distance as heuristic

### Top-5 Voting
- **What:** Voting mechanism for robustness
- **Why:** Single best match can be noisy
- **How:** Takes majority vote of 5 nearest neighbors

## 🎁 Bonus Features

- ✅ Batch image processing
- ✅ Index statistics
- ✅ Travel time estimation
- ✅ Alternative route capability
- ✅ Logging and debugging
- ✅ Type hints throughout
- ✅ Error handling

## 📞 Support

1. Check `README.md` for detailed info
2. Review `API_DOCS.md` for function signatures
3. Run `examples.py` to see working code
4. Check `setup_check.py` for diagnostics

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Run `setup_check.py`
3. ✅ Run `train_index.py`
4. ✅ Test with `examples.py`
5. ✅ Deploy with `flask_api.py`

---

**You're all set!** Start with `python examples.py` to see the system in action. 🚀
