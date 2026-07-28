"""
Campus Navigation App - Main inference module
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter

# Setup paths
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

# Import modules
from config import (
    INDEX_FILE, LABELS_FILE, METADATA_FILE, TOP_K, DISTANCE_THRESHOLD,
    MODEL_NAME, DEVICE, EMBEDDING_DIM, PATHFINDING_HEURISTIC_WEIGHT, CHECKPOINT
)
from embedding_model import DINOv2Embedder
from faiss_index import FAISSIndex
from pathfinding import CampusGraph, AStarPathfinder
from data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CampusNavigationApp:
    """
    Main application class for campus navigation
    """
    
    def __init__(self, index_dir: Path = None):
        """
        Initialize the campus navigation app
        
        Args:
            index_dir: Directory containing FAISS index files (default: BASE_DIR)
        """
        self.index_dir = index_dir or BASE_DIR
        self.embedder = None
        self.faiss_index = None
        self.pathfinder = None
        self.data_loader = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        logger.info("Initializing Campus Navigation App...")

        # Load fine-tuned ViT-B/14 embedder (timm, 768-d, 99.1% val_acc)
        logger.info(f"Loading DINOv2Embedder from {CHECKPOINT}")
        self.embedder = DINOv2Embedder(checkpoint_path=CHECKPOINT, device=DEVICE)
        
        # Load FAISS index
        index_path = self.index_dir / "campus.index"
        labels_path = self.index_dir / "labels.npy"
        metadata_path = self.index_dir / "metadata.json"
        
        if not index_path.exists() or not labels_path.exists():
            logger.error(f"Index files not found! Please run 'python train_index.py' first.")
            logger.error(f"Expected: {index_path} and {labels_path}")
            raise FileNotFoundError("FAISS index files not found")
        
        logger.info(f"Loading FAISS index from {index_path}")
        self.faiss_index = FAISSIndex.load(index_path, labels_path, metadata_path)
        
        # Initialize pathfinder
        logger.info("Initializing pathfinder...")
        campus_graph = CampusGraph()
        self.pathfinder = AStarPathfinder(campus_graph)
        
        # Initialize data loader
        self.data_loader = DataLoader(BASE_DIR / "dataset")
        
        logger.info("Campus Navigation App initialized successfully!")
    
    def identify_location(self, image_path: str, use_voting: bool = True) -> Dict:
        """
        Identify the location of a given image using top-5 voting mechanism
        
        Args:
            image_path: Path to the query image
            use_voting: Whether to use voting mechanism or return single best match
            
        Returns:
            Dictionary containing:
                - location: Identified location name
                - confidence: Confidence score (0-1)
                - top_5_matches: Top 5 nearest neighbors
                - distances: Distances to top 5 neighbors
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return {"error": f"Image not found: {image_path}"}
            
            # Generate embedding for query image
            logger.info(f"Generating embedding for {image_path.name}...")
            query_embedding = self.embedder.get_embedding(image_path)
            
            if query_embedding is None:
                logger.error(f"Failed to generate embedding for {image_path}")
                return {"error": "Failed to generate embedding"}
            
            # Search in FAISS index
            logger.info(f"Searching in FAISS index (top-{TOP_K})...")
            distances, neighbors = self.faiss_index.search(query_embedding, k=TOP_K)
            
            result = {
                "image_path": str(image_path),
                "top_5_matches": neighbors[:5],
                "distances": distances[:5].tolist(),
                "raw_distances": distances[:5].tolist(),
            }
            
            # Apply voting mechanism
            if use_voting:
                # Count votes for each location
                vote_counts = Counter(neighbors[:5])
                identified_location = vote_counts.most_common(1)[0][0]
                confidence = vote_counts.most_common(1)[0][1] / 5  # 0-1 scale
                
                logger.info(f"Voting results: {dict(vote_counts)}")
                logger.info(f"Identified location: {identified_location} (confidence: {confidence:.2f})")
                
                result["location"] = identified_location
                result["confidence"] = float(confidence)
                result["voting_results"] = dict(vote_counts)
            else:
                # Use single best match
                identified_location = neighbors[0]
                confidence = 1.0 / (1.0 + distances[0] / 10.0)  # Simple confidence based on distance
                
                logger.info(f"Single best match: {identified_location} (distance: {distances[0]:.4f})")
                
                result["location"] = identified_location
                result["confidence"] = float(confidence)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in identify_location: {e}", exc_info=True)
            return {"error": str(e)}
    
    def get_navigation_path(self, start_location: str, end_location: str) -> Dict:
        """
        Get navigation path from start to destination location
        
        Args:
            start_location: Starting location name
            end_location: Destination location name
            
        Returns:
            Dictionary containing:
                - path: List of locations from start to end
                - total_distance: Total distance in meters
                - estimated_time: Estimated travel time in seconds
                - error: Error message if path not found
        """
        try:
            logger.info(f"Finding path from {start_location} to {end_location}...")
            
            # Find path using A*
            path = self.pathfinder.find_path(
                start_location, 
                end_location,
                heuristic_weight=PATHFINDING_HEURISTIC_WEIGHT
            )
            
            if path is None:
                logger.warning(f"No path found from {start_location} to {end_location}")
                return {
                    "error": f"No path found from {start_location} to {end_location}",
                    "start": start_location,
                    "end": end_location
                }
            
            # Calculate distance and time
            total_distance = 0
            for i in range(len(path) - 1):
                current = path[i]
                next_loc = path[i + 1]
                
                neighbors = self.pathfinder.graph.get_neighbors(current)
                for neighbor, distance in neighbors:
                    if neighbor == next_loc:
                        total_distance += distance
                        break
            
            estimated_time = self.pathfinder.estimate_travel_time(path)
            
            logger.info(f"Path found: {' -> '.join(path)}")
            logger.info(f"Total distance: {total_distance}m, Estimated time: {estimated_time:.0f}s")
            
            return {
                "start": start_location,
                "end": end_location,
                "path": path,
                "num_locations": len(path),
                "total_distance_meters": total_distance,
                "estimated_time_seconds": estimated_time,
                "estimated_time_minutes": estimated_time / 60
            }
        
        except Exception as e:
            logger.error(f"Error in get_navigation_path: {e}", exc_info=True)
            return {"error": str(e)}
    
    def full_navigation_query(self, image_path: str, destination: str) -> Dict:
        """
        Complete navigation query: identify location and get path to destination
        
        Args:
            image_path: Path to query image
            destination: Destination location name
            
        Returns:
            Dictionary containing location identification and navigation path
        """
        result = {}
        
        # Step 1: Identify current location
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Identifying Current Location")
        logger.info("="*60)
        location_result = self.identify_location(image_path)
        result["location_identification"] = location_result
        
        if "error" in location_result:
            return result
        
        current_location = location_result["location"]
        
        # Step 2: Get navigation path
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Getting Navigation Path")
        logger.info("="*60)
        path_result = self.get_navigation_path(current_location, destination)
        result["navigation"] = path_result
        
        logger.info("\n" + "="*60)
        logger.info("FULL NAVIGATION QUERY COMPLETED")
        logger.info("="*60)
        
        return result
    
    def identify_from_pil(self, pil_image, use_voting: bool = True) -> Dict:
        """Identify location from a PIL Image (in-memory, no disk I/O).

        Uses the fine-tuned 12-class classifier when available;
        falls back to DINOv2 + FAISS for the original 6 classes.
        """
        try:
            # Embed with fine-tuned ViT-B/14 → FAISS cosine retrieval
            query_embedding = self.embedder.get_embedding_from_pil(pil_image)
            if query_embedding is None:
                return {"error": "Failed to generate embedding"}

            distances, neighbors = self.faiss_index.search(query_embedding, k=TOP_K)
            result = {
                "top_5_matches": neighbors[:5],
                "distances":     distances[:5].tolist(),
                "model":         "finetuned_faiss",
            }

            if use_voting:
                vote_counts = Counter(neighbors[:5])
                identified_location = vote_counts.most_common(1)[0][0]
                confidence = vote_counts.most_common(1)[0][1] / 5
                result["location"]       = identified_location
                result["confidence"]     = float(confidence)
                result["voting_results"] = dict(vote_counts)
            else:
                identified_location = neighbors[0]
                confidence = float(distances[0])  # cosine similarity (0→1)
                result["location"]   = identified_location
                result["confidence"] = confidence

            return result

        except Exception as e:
            logger.error(f"Error in identify_from_pil: {e}", exc_info=True)
            return {"error": str(e)}

    def get_available_locations(self) -> List[str]:
        """
        Get list of all available campus locations

        Returns:
            List of location names
        """
        locations = self.data_loader.get_all_locations(split="train")
        logger.info(f"Available locations: {locations}")
        return locations
    
    def get_index_statistics(self) -> Dict:
        """
        Get statistics about the FAISS index
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "total_indexed_items": self.faiss_index.get_total_items(),
            "embedding_dimension": self.faiss_index.embedding_dim,
            "locations": list(set(self.faiss_index.labels)),
            "index_file": str(INDEX_FILE),
            "labels_file": str(LABELS_FILE),
        }
        
        # Count items per location
        location_counts = Counter(self.faiss_index.labels)
        stats["items_per_location"] = dict(location_counts)
        
        logger.info(f"Index statistics: {stats}")
        return stats


def main():
    """
    Example usage of the Campus Navigation App
    """
    logger.info("Starting Campus Navigation App Demo...")
    
    # Initialize app
    app = CampusNavigationApp()
    
    # Get available locations
    logger.info("\nAvailable locations:")
    locations = app.get_available_locations()
    for loc in locations:
        logger.info(f"  - {loc}")
    
    # Get index statistics
    logger.info("\nIndex statistics:")
    stats = app.get_index_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Example 1: Identify a test image
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Identify Location from Image")
    logger.info("="*60)
    
    test_dir = BASE_DIR / "dataset" / "test"
    if test_dir.exists():
        # Find first image in test set
        for location_dir in test_dir.iterdir():
            if location_dir.is_dir():
                images = list(location_dir.glob("*.jpg")) + list(location_dir.glob("*.png"))
                if images:
                    test_image = images[0]
                    logger.info(f"Using test image: {test_image}")
                    
                    result = app.identify_location(str(test_image))
                    logger.info(f"Result: {result}")
                    break
    
    # Example 2: Get navigation path
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Get Navigation Path")
    logger.info("="*60)
    
    if len(locations) >= 2:
        start = locations[0]
        end = locations[1]
        logger.info(f"Finding path from {start} to {end}")
        
        result = app.get_navigation_path(start, end)
        logger.info(f"Result: {result}")
    
    logger.info("\nDemo completed!")


if __name__ == "__main__":
    main()
