"""
Data loading utilities for image processing
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads images and metadata from campus dataset
    """
    
    def __init__(self, dataset_dir: Path, image_extensions: set = None):
        """
        Initialize data loader
        
        Args:
            dataset_dir: Root directory of dataset
            image_extensions: Set of valid image extensions
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_extensions = image_extensions or {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    def get_images_by_location(self, split: str = "train") -> Tuple[List[Path], List[str]]:
        """
        Load all images organized by location
        
        Args:
            split: Dataset split ("train" or "test")
            
        Returns:
            Tuple of (image_paths, labels) where labels are location names
        """
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            logger.error(f"Dataset directory not found: {split_dir}")
            return [], []
        
        image_paths = []
        labels = []
        
        # Iterate through location directories
        for location_dir in split_dir.iterdir():
            if not location_dir.is_dir():
                continue
            
            location_name = location_dir.name
            logger.info(f"Processing location: {location_name}")
            
            # Find all images in location directory
            location_images = []
            for ext in self.image_extensions:
                location_images.extend(location_dir.glob(f"*{ext}"))
                location_images.extend(location_dir.glob(f"*{ext.upper()}"))
            
            # Add to collections
            for image_path in sorted(location_images):
                image_paths.append(image_path)
                labels.append(location_name)
            
            logger.info(f"Found {len(location_images)} images in {location_name}")
        
        logger.info(f"Total images loaded: {len(image_paths)}")
        return image_paths, labels
    
    def get_images_by_location_dict(self, split: str = "train") -> dict:
        """
        Get images organized as dictionary {location: [image_paths]}
        
        Args:
            split: Dataset split ("train" or "test")
            
        Returns:
            Dictionary mapping location names to lists of image paths
        """
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            logger.error(f"Dataset directory not found: {split_dir}")
            return {}
        
        location_images = {}
        
        for location_dir in split_dir.iterdir():
            if not location_dir.is_dir():
                continue
            
            location_name = location_dir.name
            images = []
            
            for ext in self.image_extensions:
                images.extend(location_dir.glob(f"*{ext}"))
                images.extend(location_dir.glob(f"*{ext.upper()}"))
            
            location_images[location_name] = sorted(images)
            logger.info(f"Found {len(images)} images in {location_name}")
        
        return location_images
    
    def get_all_locations(self, split: str = "train") -> List[str]:
        """
        Get all location names in dataset
        
        Args:
            split: Dataset split ("train" or "test")
            
        Returns:
            List of location names
        """
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            logger.error(f"Dataset directory not found: {split_dir}")
            return []
        
        locations = [d.name for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(locations)} locations: {locations}")
        return sorted(locations)
    
    def count_images(self, split: str = "train") -> dict:
        """
        Count images per location
        
        Args:
            split: Dataset split ("train" or "test")
            
        Returns:
            Dictionary mapping location names to image counts
        """
        image_dict = self.get_images_by_location_dict(split)
        counts = {loc: len(images) for loc, images in image_dict.items()}
        
        total = sum(counts.values())
        logger.info(f"Image counts per location:\n" + 
                   "\n".join(f"  {loc}: {count}" for loc, count in sorted(counts.items())))
        logger.info(f"Total images: {total}")
        
        return counts
