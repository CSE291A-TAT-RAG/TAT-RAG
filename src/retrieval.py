"""Retrieval and generation module for RAG pipeline."""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient

from .config import RAGConfig
from .llm_providers import create_llm_provider
from .embedding_providers import create_embedding_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG pipeline.

        Args:
            config: RAG configuration object
        """
        self.config = config
        self.qdrant_client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )

        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            provider_type=config.llm.provider,
            model_name=config.llm.model_name,
            api_key=config.llm.api_key,
            base_url=config.llm.ollama_base_url
        )

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(
            provider_type=config.embedding.provider,
            model_name=config.embedding.model_name,
            api_key=config.embedding.api_key,
            device=config.embedding.device,
            cache_dir=config.embedding.cache_dir
        )

        # Set vector size dynamically if not set
        if config.qdrant.vector_size is None:
            config.qdrant.vector_size = self.embedding_provider.get_dimension()
            logger.info(f"Set vector size to {config.qdrant.vector_size} based on embedding model")

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        return self.embedding_provider.embed_query(query)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Qdrant.

        Args:
            query: Query string
            top_k: Number of documents to retrieve (default from config)

        Returns:
            List of retrieved documents with content, metadata, and score
        """
        if top_k is None:
            top_k = self.config.top_k

        query_vector = self.embed_query(query)

        search_result = self.qdrant_client.search(
            collection_name=self.config.qdrant.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        retrieved_docs = []
        for hit in search_result:
            retrieved_docs.append({
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score,
                "id": hit.id
            })

        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        return retrieved_docs

    def generate_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Generate a prompt for the LLM with query and retrieved contexts.

        Args:
            query: User query
            contexts: List of retrieved context strings

        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided contexts.

{context_text}

Question: {query}

Answer: Provide a detailed and accurate answer based on the contexts above. If the contexts don't contain enough information to answer the question, say so."""

        return prompt

    def generate(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Generate an answer using the LLM.

        Args:
            query: User query
            contexts: List of retrieved context strings

        Returns:
            Dictionary with answer and metadata
        """
        prompt = self.generate_prompt(query, contexts)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided contexts."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_provider.generate(
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )

        logger.info(f"Generated answer for query: {query[:50]}...")

        return {
            "answer": response["content"],
            "query": query,
            "contexts": contexts,
            "model": response["model"],
            "usage": response["usage"]
        }

    def query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.

        Args:
            query: User query
            top_k: Number of documents to retrieve (default from config)

        Returns:
            Dictionary with answer, contexts, and metadata
        """
        logger.info(f"Processing query: {query}")

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        contexts = [doc["content"] for doc in retrieved_docs]

        # Generate answer
        result = self.generate(query, contexts)

        # Add retrieved documents info
        result["retrieved_docs"] = retrieved_docs

        return result

    def batch_query(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query

        Returns:
            List of result dictionaries
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.query(query, top_k)
            results.append(result)

        return results
