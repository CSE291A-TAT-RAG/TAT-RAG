"""Data ingestion module for processing and storing documents in Qdrant."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import RAGConfig
from .embedding_providers import create_embedding_provider
from .parsers import create_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handle document loading, chunking, embedding, and storage."""

    def __init__(self, config: RAGConfig):
        """
        Initialize the ingestion pipeline.

        Args:
            config: RAG configuration object
        """
        self.config = config
        self.qdrant_client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(
            provider_type=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            cache_dir=config.embedding.cache_dir
        )

        # Set vector size dynamically from embedding provider
        if config.qdrant.vector_size is None:
            config.qdrant.vector_size = self.embedding_provider.get_dimension()
            logger.info(f"Set vector size to {config.qdrant.vector_size} based on embedding model")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.qdrant.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.config.qdrant.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.config.qdrant.collection_name,
                vectors_config=VectorParams(
                    size=self.config.qdrant.vector_size,
                    distance=Distance.COSINE
                ),
            )
        else:
            logger.info(f"Collection {self.config.qdrant.collection_name} already exists")

    def load_documents(
        self,
        path: str,
        file_type: str = "txt",
        parser_type: str = "langchain"
    ) -> List[Dict[str, Any]]:
        """
        Load documents from a file or directory using specified parser.

        Args:
            path: Path to file or directory
            file_type: Type of files to load (txt, pdf)
            parser_type: Parser to use ("langchain" or "fitz")
                - "langchain": Simple, fast loading (default)
                - "fitz": Advanced PDF parsing with better text extraction

        Returns:
            List of document dictionaries with content and metadata
        """
        # Create parser instance
        parser = create_parser(parser_type)

        # Parse documents
        documents = parser.parse(path, file_type=file_type)

        logger.info(
            f"Loaded {len(documents)} documents from {path} "
            f"using {parser_type} parser"
        )
        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunked document dictionaries
        """
        chunks = []
        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": i,
                        "total_chunks": len(text_chunks)
                    }
                })

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = self.config.embedding.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_provider.embed_texts(batch)
            embeddings.extend(batch_embeddings)
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        return embeddings

    def _generate_chunk_id(self, source: str, chunk_id: int) -> str:
        """
        Generate a deterministic ID based on source path and chunk index.

        Args:
            source: Source file path
            chunk_id: Chunk index within the document

        Returns:
            Deterministic UUID string
        """
        # Create a deterministic ID from source path and chunk_id
        unique_string = f"{source}:{chunk_id}"
        hash_object = hashlib.md5(unique_string.encode())
        # Convert hash to UUID format
        return str(uuid.UUID(hash_object.hexdigest()))

    def store_chunks(self, chunks: List[Dict[str, Any]], overwrite: bool = True) -> int:
        """
        Store chunked documents with embeddings in Qdrant.

        Args:
            chunks: List of chunked document dictionaries
            overwrite: If True, delete existing chunks from the same source before storing

        Returns:
            Number of chunks stored
        """
        # If overwrite mode, delete existing chunks from the same sources
        if overwrite and chunks:
            sources = set(chunk["metadata"].get("source") for chunk in chunks)
            logger.info(f"Overwrite mode enabled. Sources to check: {sources}")
            for source in sources:
                if source:
                    logger.info(f"Deleting existing chunks from source: {source}")
                    result = self.qdrant_client.delete(
                        collection_name=self.config.qdrant.collection_name,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.source",
                                    match=MatchValue(value=source)
                                )
                            ]
                        )
                    )
                    logger.info(f"Deletion completed for source: {source}")

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate deterministic ID based on source and chunk_id
            source = chunk["metadata"].get("source", "unknown")
            chunk_id = chunk["metadata"].get("chunk_id", 0)
            point_id = self._generate_chunk_id(source, chunk_id)

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.config.qdrant.collection_name,
                points=batch
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

        logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
        return len(points)

    def ingest(
        self,
        path: str,
        file_type: str = "txt",
        parser_type: str = "langchain",
        overwrite: bool = True
    ) -> int:
        """
        Complete ingestion pipeline: load, chunk, embed, and store.

        Args:
            path: Path to file or directory
            file_type: Type of files to load (txt, pdf)
            parser_type: Parser to use ("langchain" or "fitz")
                - "langchain": Simple, fast loading (default)
                - "fitz": Advanced PDF parsing for financial reports
            overwrite: If True, replace existing chunks from the same source (default: True)

        Returns:
            Number of chunks stored
        """
        logger.info(
            f"Starting ingestion pipeline for: {path} "
            f"(parser={parser_type}, overwrite={overwrite})"
        )
        documents = self.load_documents(path, file_type, parser_type)
        chunks = self.chunk_documents(documents)
        count = self.store_chunks(chunks, overwrite=overwrite)
        logger.info(f"Ingestion complete. Stored {count} chunks.")
        return count
