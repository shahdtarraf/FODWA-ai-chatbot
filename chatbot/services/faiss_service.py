"""
FAISS Service — Singleton with lazy loading.
Loads index.faiss and chunks.json on first request only.
Uses IndexFlatL2, returns top_k=3 results.

PRESERVED EXACTLY from FastAPI version — no changes needed.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Path to data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")


class FAISSService:
    """Singleton FAISS service with lazy loading."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.index = None
        self.chunks = None
        self._initialized = True
        logger.info("FAISSService singleton created (not loaded yet)")

    def _load_index(self):
        """
        Load FAISS index and chunks from disk.
        Called lazily on first search request.
        """
        try:
            import faiss

            if not os.path.exists(INDEX_PATH):
                logger.critical(f"FAISS index not found at: {INDEX_PATH}")
                return False

            if not os.path.exists(CHUNKS_PATH):
                logger.critical(f"Chunks file not found at: {CHUNKS_PATH}")
                return False

            # Load FAISS index
            self.index = faiss.read_index(INDEX_PATH)
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

            # Load text chunks
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(f"Chunks loaded: {len(self.chunks)} chunks")

            # Validate consistency
            if self.index.ntotal != len(self.chunks):
                logger.warning(
                    f"Mismatch: index has {self.index.ntotal} vectors "
                    f"but chunks.json has {len(self.chunks)} entries"
                )

            return True

        except Exception as e:
            logger.critical(f"Failed to load FAISS data: {e}")
            self.index = None
            self.chunks = None
            return False

    def search(self, query_embedding: list[float], top_k: int = 15) -> list[str]:
        """
        Search the FAISS index for the most similar chunks.

        Args:
            query_embedding: Query vector from OpenAI embedding.
            top_k: Number of results to return.

        Returns:
            List of text chunks most relevant to the query.
        """
        # Lazy load on first request
        if self.index is None:
            if not self._load_index():
                logger.error("FAISS data unavailable — returning empty results")
                return []

        try:
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search
            distances, indices = self.index.search(query_vector, top_k)

            # Collect matching chunks
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    results.append(self.chunks[idx])
                else:
                    logger.warning(f"FAISS returned invalid index: {idx}")

            logger.info(f"FAISS search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []


# Global singleton instance
faiss_service = FAISSService()
