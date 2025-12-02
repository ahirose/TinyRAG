"""
Simple Vector Database implementation using numpy.
Supports basic operations: add, search, and delete.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle


class SimpleVectorDB:
    """A simple in-memory vector database using numpy for similarity search."""

    def __init__(self, dimension: int = 384):
        """
        Initialize the vector database.

        Args:
            dimension: The dimension of the vectors to store
        """
        self.dimension = dimension
        self.vectors = np.array([]).reshape(0, dimension)
        self.documents = []
        self.metadata = []

    def add(self, vectors: np.ndarray, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add vectors and their corresponding documents to the database.

        Args:
            vectors: numpy array of shape (n, dimension)
            documents: list of document texts
            metadata: optional list of metadata dictionaries
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match database dimension {self.dimension}")

        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")

        self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) > 0 else vectors
        self.documents.extend(documents)

        if metadata is None:
            metadata = [{}] * len(documents)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for the most similar documents to the query vector.

        Args:
            query_vector: numpy array of shape (dimension,)
            top_k: number of top results to return

        Returns:
            List of tuples (document, similarity_score, metadata)
        """
        if len(self.vectors) == 0:
            return []

        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} does not match database dimension {self.dimension}")

        # Normalize vectors for cosine similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)

        # Calculate cosine similarity
        similarities = np.dot(vectors_norm, query_norm)

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))

        return results

    def save(self, filepath: str):
        """Save the database to a file."""
        data = {
            'dimension': self.dimension,
            'vectors': self.vectors,
            'documents': self.documents,
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load the database from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.dimension = data['dimension']
        self.vectors = data['vectors']
        self.documents = data['documents']
        self.metadata = data['metadata']

    def clear(self):
        """Clear all data from the database."""
        self.vectors = np.array([]).reshape(0, self.dimension)
        self.documents = []
        self.metadata = []

    def __len__(self):
        """Return the number of documents in the database."""
        return len(self.documents)
