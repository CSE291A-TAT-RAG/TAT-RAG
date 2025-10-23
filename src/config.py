"""Configuration settings for the RAG pipeline."""

import os
from dataclasses import dataclass
from typing import Optional


def _get_bool_env(var_name: str, default: bool = False) -> bool:
    """Read boolean flag from environment."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")
    vector_size: Optional[int] = None  # Will be set dynamically based on embedding model
    distance_metric: str = "Cosine"
    timeout: int = int(os.getenv("QDRANT_TIMEOUT", "60"))


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    # Provider: 'ollama', 'bedrock', or 'gemini'
    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    model_name: str = os.getenv("LLM_MODEL", "qwen2.5:8b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    request_interval: float = float(os.getenv("LLM_REQUEST_INTERVAL", "0"))

    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # AWS Bedrock specific
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = os.getenv("AWS_SESSION_TOKEN")
    aws_profile_name: Optional[str] = os.getenv("AWS_PROFILE_NAME")

    # Gemini specific
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    # Provider: 'local' only
    provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # Local model specific
    device: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)  # 'cuda', 'cpu', or None for auto
    cache_dir: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR", None)  # Local model cache directory


@dataclass
class RAGASEvaluatorConfig:
    """RAGAS evaluation settings."""
    timeout: int = int(os.getenv("RAGAS_TIMEOUT", "180"))
    max_workers: int = int(os.getenv("RAGAS_MAX_WORKERS", "2"))
    max_retries: int = int(os.getenv("RAGAS_MAX_RETRIES", "3"))
    max_wait: int = int(os.getenv("RAGAS_MAX_WAIT", "30"))


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))  # Minimum similarity score
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    hybrid_search: bool = _get_bool_env("HYBRID_SEARCH_ENABLED", False)

    qdrant: QdrantConfig = None
    llm: LLMConfig = None
    embedding: EmbeddingConfig = None
    ragas: RAGASEvaluatorConfig = None

    def __post_init__(self):
        if self.qdrant is None:
            self.qdrant = QdrantConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.ragas is None:
            self.ragas = RAGASEvaluatorConfig()


# Default configuration instance
config = RAGConfig()
