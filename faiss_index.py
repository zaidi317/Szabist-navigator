"""
FAISS indexing and searching for image embeddings
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class FAISSIndex:
    """
    Manages FAISS index for embedding-based image search
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings (384 for ViT-S/14)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.labels = []
        self.metadata = {
            "location": [],
            "image_path": []
        }
    
    def add_embeddings(self, embeddings: np.ndarray, labels: List[str]):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Array of shape (N, embedding_dim) containing embeddings
            labels: List of labels corresponding to embeddings (location names)
        """
        if embeddings.shape[0] != len(labels):
            raise ValueError("Number of embeddings must match number of labels")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        self.labels.extend(labels)
        
        logger.info(f"Added {len(labels)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Search for nearest neighbors
        
        Args:
            query_embedding: Query embedding (1D array, shape: (embedding_dim,))
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, labels) for top-k nearest neighbors
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Reshape to (1, embedding_dim) for search
        query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        # Extract labels for returned indices
        result_labels = [self.labels[idx] for idx in indices[0]]
        
        return distances[0], result_labels
    
    def save(self, index_path: Path, labels_path: Path, metadata_path: Path = None):
        """
        Save index and labels to disk
        
        Args:
            index_path: Path to save FAISS index
            labels_path: Path to save labels (numpy file)
            metadata_path: Path to save metadata (optional)
        """
        # Ensure parent directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save labels
        np.save(labels_path, np.array(self.labels))
        logger.info(f"Saved labels to {labels_path}")
        
        # Save metadata if provided
        if metadata_path:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
    
    @staticmethod
    def load(index_path: Path, labels_path: Path, metadata_path: Path = None) -> 'FAISSIndex':
        """
        Load index and labels from disk
        
        Args:
            index_path: Path to FAISS index file
            labels_path: Path to labels file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            FAISSIndex object
        """
        # Load FAISS index
        faiss_index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load labels
        labels = np.load(labels_path, allow_pickle=True).tolist()
        logger.info(f"Loaded {len(labels)} labels from {labels_path}")
        
        # Create FAISSIndex object
        embedding_dim = faiss_index.d
        faiss_obj = FAISSIndex(embedding_dim)
        faiss_obj.index = faiss_index
        faiss_obj.labels = labels
        
        # Load metadata if provided
        if metadata_path and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                faiss_obj.metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        
        return faiss_obj
    
    def get_total_items(self) -> int:
        """Get total number of indexed items"""
        return self.index.ntotal
