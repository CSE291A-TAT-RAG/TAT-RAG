"""Data ingestion module for processing and storing documents in Qdrant."""

import logging
from typing import List, Dict, Any
import uuid
import hashlib
import json
import zipfile
import io
import csv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .config import RAGConfig
from .embedding_providers import create_embedding_provider

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

    def load_chunks_from_zip(self, zip_path: str) -> List[Dict[str, Any]]:
        """
        Load pre-created chunks from a zip archive containing JSONL metadata and CSV tables.

        Args:
            zip_path: Path to the zip archive

        Returns:
            List of chunk dictionaries compatible with downstream storage
        """
        logger.info(f"Loading pre-chunked data from zip archive: {zip_path}")

        chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}

        with zipfile.ZipFile(zip_path) as archive:
            try:
                jsonl_entry = archive.open("chunks_all.jsonl")
            except KeyError as exc:
                raise FileNotFoundError("Expected 'chunks_all.jsonl' inside archive") from exc

            with jsonl_entry:
                for raw_line in jsonl_entry:
                    if not raw_line.strip():
                        continue
                    record = json.loads(raw_line.decode("utf-8-sig"))
                    metadata = dict(record.get("metadata", {}))
                    doc_id = metadata.get("doc_id", "unknown")
                    content = record.get("text", "")

                    metadata.setdefault("source", doc_id)
                    metadata.setdefault("parser", "prechunked")

                    if metadata.get("type") == "table" and content:
                        table_path = content
                        metadata["table_path"] = table_path
                        try:
                            with archive.open(table_path) as table_file:
                                table_text = io.TextIOWrapper(table_file, encoding="utf-8", newline="")
                                reader = csv.reader(table_text)
                                rows = [" | ".join(cell.strip() for cell in row) for row in reader]
                                # Drop empty rows to keep embeddings focused on data
                                rows = [row for row in rows if row]
                                content = "\n".join(rows)
                        except KeyError:
                            logger.warning("Missing table CSV '%s' referenced in JSONL", table_path)
                            content = ""

                    chunk = {
                        "content": self._add_metadata_context(content, metadata, doc_id),
                        "metadata": metadata
                    }
                    chunks_by_doc.setdefault(doc_id, []).append(chunk)

        chunks: List[Dict[str, Any]] = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            total = len(doc_chunks)
            for index, chunk in enumerate(doc_chunks):
                metadata = dict(chunk["metadata"])
                metadata["chunk_id"] = index
                metadata["total_chunks"] = total
                metadata.setdefault("source", doc_id)
                chunks.append({
                    "content": chunk["content"],
                    "metadata": metadata
                })

        logger.info(
            "Loaded %d chunks across %d documents from %s",
            len(chunks),
            len(chunks_by_doc),
            zip_path
        )
        return chunks

    def _add_metadata_context(self, content: str, metadata: Dict[str, Any], doc_id: str) -> str:
        """
        Prefix raw chunk content with key metadata to improve retrievability.

        Args:
            content: Original chunk text
            metadata: Chunk metadata dictionary
            doc_id: Document identifier the chunk belongs to

        Returns:
            Enriched content string including metadata cues
        """
        context_parts = []

        source = metadata.get("source") or doc_id
        if source:
            context_parts.append(f"Document: {source}")

        section = metadata.get("section_path")
        if section:
            context_parts.append(f"Section: {section}")

        page = metadata.get("page")
        if page is not None:
            context_parts.append(f"Page: {page}")

        chunk_type = metadata.get("type")
        if chunk_type:
            context_parts.append(f"Content type: {chunk_type}")

        if not context_parts:
            return content

        context_header = " | ".join(str(part) for part in context_parts)

        if content:
            return f"{context_header}\n\n{content}"

        return context_header

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
        overwrite: bool = True
    ) -> int:
        """
        Complete ingestion pipeline: load, chunk, embed, and store.

        Args:
            path: Path to zip archive containing pre-chunked data
            overwrite: If True, replace existing chunks from the same source (default: True)

        Returns:
            Number of chunks stored
        """
        logger.info(
            f"Starting ingestion pipeline for: {path} "
            f"(overwrite={overwrite})"
        )
        if not path.lower().endswith(".zip"):
            raise ValueError("Only zip archives containing pre-chunked data are supported. Please provide a .zip file.")
        chunks = self.load_chunks_from_zip(path)
        count = self.store_chunks(chunks, overwrite=overwrite)
        logger.info(f"Ingestion complete. Stored {count} chunks.")
        return count
