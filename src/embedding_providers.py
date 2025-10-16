"""Embedding provider abstraction layer supporting OpenAI and local models."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import warnings

from openai import OpenAI

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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., text-embedding-ada-002, text-embedding-3-small)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        # Set dimension based on model
        if "ada-002" in model_name:
            self.dimension = 1536
        elif "text-embedding-3-small" in model_name:
            self.dimension = 1536
        elif "text-embedding-3-large" in model_name:
            self.dimension = 3072
        else:
            self.dimension = 1536  # default

        logger.info(f"Initialized OpenAI embedding provider with model: {model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        response = self.client.embeddings.create(
            input=query,
            model=self.model_name
        )
        return response.data[0].embedding

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


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
        provider_type: Type of provider ('openai' or 'local')
        model_name: Model name
        api_key: API key (for OpenAI)
        device: Device (for local models)
        cache_dir: Cache directory (for local models)

    Returns:
        EmbeddingProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires an API key")
        return OpenAIEmbeddingProvider(api_key=api_key, model_name=model_name)

    elif provider_type == "local":
        return LocalEmbeddingProvider(model_name=model_name, device=device, cache_dir=cache_dir)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'openai' or 'local'")
