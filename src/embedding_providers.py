"""Embedding provider abstraction layer for local models."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import warnings

# Suppress FutureWarnings from sentence-transformers
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize local embedding provider.

        Args:
            model_name: HuggingFace model name (e.g., BAAI/bge-m3, BAAI/bge-small-zh-v1.5)
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Local directory to cache models (e.g., './models')
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        # Log cache directory info
        if cache_dir:
            logger.info(f"Using local model cache directory: {cache_dir}")
        else:
            logger.info("Using default HuggingFace cache directory")

        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

        # Get dimension from model
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Initialized local embedding provider with model: {model_name} "
                   f"(dimension: {self.dimension}, device: {self.model.device})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents to maintain compatibility."""
        return self.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


def create_embedding_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> EmbeddingProvider:
    """
    Factory function to create embedding provider.

    Args:
        provider_type: Type of provider ('local' only)
        model_name: Model name
        api_key: API key (deprecated, not used)
        device: Device (for local models)
        cache_dir: Cache directory (for local models)

    Returns:
        EmbeddingProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "local":
        return LocalEmbeddingProvider(model_name=model_name, device=device, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'local'")
