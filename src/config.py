"""Configuration settings for the RAG pipeline."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")
    vector_size: Optional[int] = None  # Will be set dynamically based on embedding model
    distance_metric: str = "Cosine"


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    # Provider: 'openai' or 'ollama'
    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    model_name: str = os.getenv("LLM_MODEL", "qwen2.5:8b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # OpenAI specific
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    # Provider: 'openai' or 'local'
    provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # OpenAI specific
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Local model specific
    device: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)  # 'cuda', 'cpu', or None for auto
    cache_dir: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR", None)  # Local model cache directory


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    qdrant: QdrantConfig = None
    llm: LLMConfig = None
    embedding: EmbeddingConfig = None

    def __post_init__(self):
        if self.qdrant is None:
            self.qdrant = QdrantConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()


# Default configuration instance
config = RAGConfig()
