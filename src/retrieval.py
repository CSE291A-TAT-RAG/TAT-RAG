"""Retrieval and generation module for RAG pipeline."""

import logging
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient

from .config import RAGConfig
from .llm_providers import create_llm_provider
from qdrant_client import models as qdrant_models

from .embedding_providers import create_embedding_provider
from .rerankers import create_reranker

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
        configured_name = getattr(config.qdrant, "dense_vector_name", None)
        if isinstance(configured_name, str):
            configured_name = configured_name.strip() or None
        self._configured_dense_vector_name: Optional[str] = configured_name
        self._dense_prefetch_using: Optional[str] = None
        self._checked_dense_vector_support: bool = False

        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            provider_type=config.llm.provider,
            model_name=config.llm.model_name,
            api_key=config.llm.gemini_api_key,
            base_url=config.llm.ollama_base_url,
            region_name=config.llm.aws_region,
            aws_access_key_id=config.llm.aws_access_key_id,
            aws_secret_access_key=config.llm.aws_secret_access_key,
            aws_session_token=config.llm.aws_session_token,
            aws_profile_name=config.llm.aws_profile_name,
            request_interval=config.llm.request_interval
        )

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(
            provider_type=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            cache_dir=config.embedding.cache_dir
        )
        self.reranker = create_reranker(config.rerank)

        # Set vector size dynamically if not set
        if config.qdrant.vector_size is None:
            config.qdrant.vector_size = self.embedding_provider.get_dimension()
            logger.info(f"Set vector size to {config.qdrant.vector_size} based on embedding model")

        self._ensure_dense_prefetch_vector()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        return self.embedding_provider.embed_query(query)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search with optional auto-filtering.

        Args:
            query: Query string
            top_k: Maximum number of documents to retrieve (default from config)
            score_threshold: Minimum similarity score (default from config)

        Returns:
            List of retrieved documents with content, metadata, and score
        """
        if top_k is None:
            top_k = self.config.top_k
        if score_threshold is None:
            score_threshold = self.config.score_threshold

        query_vector = self.embed_query(query)
        self._ensure_dense_prefetch_vector()
        dense_query_vector = self._make_query_vector(query_vector)

        if self.config.hybrid_search and query.strip():
            logger.info("Performing Hybrid (Dense + BM25) search.")
            try:
                hybrid_prefetch = max(top_k, getattr(self.config, "hybrid_prefetch", top_k))
                self._ensure_dense_prefetch_vector()

                query_response = None
                for attempt in range(2):
                    dense_prefetch_kwargs = dict(
                        query=query_vector,
                        limit=hybrid_prefetch,
                        score_threshold=score_threshold,
                    )
                    if self._dense_prefetch_using:
                        dense_prefetch_kwargs["using"] = self._dense_prefetch_using

                    dense_prefetch = qdrant_models.Prefetch(**dense_prefetch_kwargs)

                    text_filter = qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="content",
                                match=qdrant_models.MatchText(text=query),
                            )
                        ]
                    )
                    text_prefetch = qdrant_models.Prefetch(
                        filter=text_filter,
                        limit=hybrid_prefetch,
                    )

                    fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)

                    try:
                        query_response = self.qdrant_client.query_points(
                            collection_name=self.config.qdrant.collection_name,
                            prefetch=[dense_prefetch, text_prefetch],
                            query=fusion_query,
                            limit=top_k,
                            with_payload=True,
                            with_vectors=False,
                        )
                        break
                    except Exception as hybrid_exc:
                        error_text = str(hybrid_exc)
                        if (
                            self._dense_prefetch_using
                            and "Vector with name" in error_text
                        ):
                            logger.info(
                                "Dense vector name '%s' is not configured in Qdrant; retrying without named vector.",
                                self._dense_prefetch_using,
                            )
                            self._dense_prefetch_using = None
                            self._checked_dense_vector_support = True
                            continue
                        raise hybrid_exc

                if query_response is None:
                    raise RuntimeError("Hybrid query failed without response.")

                search_result = getattr(query_response, "points", query_response)
            except Exception as exc:
                logger.warning(
                    "Hybrid query_points failed (%s); falling back to dense-only search.", exc
                )
                search_result = self.qdrant_client.search(
                    collection_name=self.config.qdrant.collection_name,
                    query_vector=dense_query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
        else:
            logger.info("Performing Dense-only vector search.")
            search_result = self.qdrant_client.search(
                collection_name=self.config.qdrant.collection_name,
                query_vector=dense_query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

        formatted_docs = self._format_retrieved_docs(search_result, query, score_threshold)
        formatted_docs = self._maybe_rerank(query, formatted_docs)
        return formatted_docs

    def _ensure_dense_prefetch_vector(self) -> None:
        """
        Ensure we only ask for a named dense vector if the collection actually exposes it.
        """
        if self._checked_dense_vector_support:
            return

        dense_name = self._configured_dense_vector_name
        if not dense_name:
            self._checked_dense_vector_support = True
            return

        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.config.qdrant.collection_name
            )
        except Exception as exc:
            logger.debug(
                "Could not inspect collection vector schema (%s); assuming unnamed vector.",
                exc,
            )
            return

        vectors_config = getattr(collection_info.config.params, "vectors", None)
        if isinstance(vectors_config, dict):
            if dense_name in vectors_config:
                self._dense_prefetch_using = dense_name
            else:
                logger.info(
                    "Configured dense vector name '%s' not found in collection schema; using default vector.",
                    dense_name,
                )
                self._dense_prefetch_using = None
        else:
            self._dense_prefetch_using = None
        self._checked_dense_vector_support = True

    def _make_query_vector(self, query_vector: List[float]):
        """
        Prepare query vector for Qdrant, using named vector when available.
        """
        if self._dense_prefetch_using:
            return qdrant_models.NamedVector(
                name=self._dense_prefetch_using,
                vector=query_vector,
            )
        return query_vector

    def _format_retrieved_docs(self, search_result, query: str, score_threshold: Optional[float]) -> List[Dict[str, Any]]:
        """
        Normalize Qdrant search results into payload dictionaries.
        """
        retrieved_docs: List[Dict[str, Any]] = []
        for hit in search_result:
            payload = hit.payload or {}
            retrieved_docs.append({
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "score": hit.score,
                "id": hit.id
            })

        logger.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}... "
            f"(threshold: {score_threshold:.2f})"
        )
        return retrieved_docs

    def _maybe_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optionally rerank retrieved documents with an external reranker.
        """
        if not documents or not self.reranker:
            return documents

        try:
            reranked = self.reranker.rerank(query, documents)
        except Exception as exc:
            logger.warning("Reranking failed for query '%s': %s", query[:50], exc, exc_info=True)
            return documents

        return reranked or documents

    def _apply_adaptive_filters(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dynamically trim low-relevance chunks and optionally cap per-document fan-out.
        """
        if not documents or not getattr(self.config, "adaptive_filter_enabled", False):
            return documents

        min_keep = max(0, getattr(self.config, "adaptive_min_keep", 0))
        score_ratio = getattr(self.config, "adaptive_score_ratio", 0.0)
        score_drop = getattr(self.config, "adaptive_score_drop", 0.0)
        max_chunks_per_doc = getattr(self.config, "max_chunks_per_doc", None)

        top_score = documents[0].get("score") or 0.0
        per_doc_counts: Dict[str, int] = defaultdict(int)
        filtered: List[Dict[str, Any]] = []

        for doc in documents:
            score = doc.get("score") or 0.0
            metadata = doc.get("metadata") or {}
            doc_key = metadata.get("source") or metadata.get("doc_id") or ""
            if max_chunks_per_doc and doc_key:
                if per_doc_counts[doc_key] >= max_chunks_per_doc:
                    continue

            keep = True
            if len(filtered) >= min_keep:
                keep_by_ratio = True
                keep_by_drop = True

                if score_ratio > 0 and top_score > 0:
                    keep_by_ratio = (score / top_score) >= score_ratio
                if score_drop > 0 and top_score >= score:
                    keep_by_drop = (top_score - score) <= score_drop

                keep = keep_by_ratio or keep_by_drop

            if not keep:
                break

            filtered.append(doc)
            if doc_key:
                per_doc_counts[doc_key] += 1

        if not filtered:
            return documents[:max(min_keep, 1)]
        return filtered

    def _select_docs_for_generation(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Slice the reranked documents according to the final context limit.
        """
        limit = getattr(self.config, "final_context_limit", None)
        if isinstance(limit, int) and limit > 0:
            return documents[:limit]
        return documents

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
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided contexts.",
            },
            {"role": "user", "content": prompt},
        ]

        max_attempts = 3
        response: Optional[Dict[str, Any]] = None
        answer: str = ""
        usage: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            response = self.llm_provider.generate(
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

            usage = dict(response.get("usage", {}) or {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
                usage["total_tokens"] = total_tokens

            answer = (response.get("content") or "").strip()

            if answer or total_tokens > 0:
                break

            logger.warning(
                "Received empty response from LLM (attempt %s/%s) for query: %s",
                attempt,
                max_attempts,
                query[:50],
            )

        else:
            logger.error(
                "LLM failed to return a non-empty response after %s attempts for query: %s",
                max_attempts,
                query[:50],
            )

        logger.info(f"Generated answer for query: {query[:50]}...")

        return {
            "answer": answer,
            "query": query,
            "contexts": contexts,
            "model": response.get("model") if response else None,
            "usage": usage,
        }


    def query_prompt(self) -> str:
        """
        Generate a system prompt for the LLM.

        Args:
            query: User query

        Returns:
            Formatted prompt string
        """

        prompt = """You are an expert financial analyst and a search query decomposition assistant. Your task is to break down a complex user question into a series of simple, self-contained sub-queries. These sub-queries will be used to retrieve information from a database of financial documents.

IMPORTANT RULES (you must follow them exactly):
1. NEVER answer the user's question.
2. NEVER perform calculations or give factual answers.
3. Decompose the question into simple, independent queries that can each be answered by a small piece of text.
4. Each sub-query should be a clear, direct question or statement about a single piece of information.
5. Preserve key entities, numbers, and dates from the original question in the sub-queries.
6. ALWAYS output ONLY a valid JSON object in the format shown below.
7. NEVER output any text, explanation, or code block markers before or after the JSON object.

### High-Quality Examples

**Example 1:**
User Question: "What is the difference in total revenue in 2019 between A10-Networks and Oracle?"

Your JSON Output:
{
  "queries": [
    "total revenue of A10-Networks in 2019",
    "total revenue of Oracle in 2019"
  ]
}

**Example 2:**
User Question: "What was the total cost of revenue for A10 Networks in 2019 and what were its main components?"

Your JSON Output:
{
  "queries": [
    "A10 Networks total cost of revenue in 2019",
    "components of A10 Networks cost of revenue"
  ]
}

### Your Task

Now, decompose the user's question following all the rules and examples.

JSON Output Format:
{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}"""

        return prompt

    def generate_query(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer using the LLM.

        Args:
            query: User query
            contexts: List of retrieved context strings

        Returns:
            Dictionary with answer and metadata
        """
        prompt = self.query_prompt()

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": f"Do NOT answer, ONLY decompose query: {query}"},
        ]

        max_attempts = 3
        response: Optional[Dict[str, Any]] = None
        answer: str = ""
        usage: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            response = self.llm_provider.generate(
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

            usage = dict(response.get("usage", {}) or {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
                usage["total_tokens"] = total_tokens

            answer = (response.get("content") or "")
            answer = answer.replace("```json\n", "")
            answer = answer.replace("\n```", "")
            logger.info(f"Respons: {response}\n")

            if answer or total_tokens > 0:
                break

            logger.warning(
                "Received empty response from LLM (attempt %s/%s) for query: %s",
                attempt,
                max_attempts,
                query[:50],
            )

        else:
            logger.error(
                "LLM failed to return a non-empty response after %s attempts for query: %s",
                max_attempts,
                query[:50],
            )

        logger.info(f"Generated answer for query: {query[:50]}...")

        queries = []
        if answer.strip():
            try:
                queries = json.loads(answer).get("queries", [])
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON from LLM for query decomposition. Response: %s",
                    answer,
                )
        return {"answer": queries}

    def decider_prompt(self) -> str:
        """
        Generate a system prompt for the LLM to decide if a query is simple or complex.
        """
        prompt = """You are a query analysis expert. Your task is to classify a user's query as either "SIMPLE" or "COMPLEX".

A "SIMPLE" query asks for a single, specific piece of information.
A "COMPLEX" query requires comparing multiple pieces of information, summarizing multiple points, or involves multiple steps.

Respond with ONLY the word "SIMPLE" or "COMPLEX". Do not provide any explanation.

### Examples

User Query: "What was the total cost of revenue for A10 Networks in 2019?"
Your Answer: SIMPLE

User Query: "What is the difference in total revenue in 2019 between A10-Networks and Oracle?"
Your Answer: COMPLEX

User Query: "Who are the members of the board of directors in Xperi?"
Your Answer: COMPLEX

User Query: "What is A10 Networks' total revenue earned by the company in 2019?"
Your Answer: SIMPLE

### Your Task

Classify the following user query.
"""
        return prompt

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        rewrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.

        Args:
            query: User query
            top_k: Maximum number of documents to retrieve (default from config)
            score_threshold: Minimum similarity score (default from config)
            rewrite: If True, perform query decomposition (default: True)

        Returns:
            Dictionary with answer, contexts, and metadata
        """
        logger.info(f"Processing query: {query}")

        # Adaptive Routing: Decide whether to use query rewrite
        if rewrite: # Only run decider if rewrite is enabled globally
            decider_messages = [
                {"role": "system", "content": self.decider_prompt()},
                {"role": "user", "content": query},
            ]
            try:
                decision_response = self.llm_provider.generate(messages=decider_messages, temperature=0.0, max_tokens=10)
                decision = (decision_response.get("content") or "").strip().upper()
                logger.info(f"Query classified as: {decision}")
                if decision != "COMPLEX":
                    rewrite = False # Override to False if query is simple
            except Exception as e:
                logger.warning(f"Query classification failed, defaulting to rewrite=True. Error: {e}")
                rewrite = True # Default to complex handling on failure

        query_list = []
        if rewrite:
            query_list = self.generate_query(query)['answer']
            if not query_list: # Fallback if decomposition fails
                query_list = [query]
            logger.info(f"New queries: \n{query_list}\n")
        else:
            query_list = [query]

        # Retrieve relevant documents with score filtering
        all_retrieved_docs = []
        for q in query_list:
            logger.info(f"Retrieving for sub-query: {q}")
            retrieved_docs = self.retrieve(q, top_k, score_threshold)
            all_retrieved_docs.extend(retrieved_docs)

        # De-duplicate and sort all retrieved documents
        unique_docs_map = {doc['id']: doc for doc in reversed(all_retrieved_docs)}
        unique_docs = sorted(unique_docs_map.values(), key=lambda d: d['score'], reverse=True)

        # Apply adaptive filtering and final selection
        filtered_docs = self._apply_adaptive_filters(unique_docs)
        final_docs_for_generation = self._select_docs_for_generation(filtered_docs)

        contexts = [doc.get("content", "") for doc in final_docs_for_generation]

        # Generate answer
        result = self.generate(query, contexts)

        # Add retrieved documents info
        result["retrieved_docs"] = unique_docs
        result["used_retrieved_docs"] = final_docs_for_generation

        return result

    def retrieve_with_rewrite(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with query decomposition, but without generation.
        This is useful for evaluating the retrieval step with query rewriting.

        Args:
            query: User query
            top_k: Maximum number of documents to retrieve per sub-query
            score_threshold: Minimum similarity score

        Returns:
            A de-duplicated and sorted list of retrieved documents.
        """
        logger.info(f"Processing query with rewrite for retrieval evaluation: {query}")

        query_list = self.generate_query(query).get('answer', [])
        if not query_list:  # Fallback if decomposition fails
            query_list = [query]
        logger.info(f"Decomposed queries for retrieval: \n{query_list}\n")

        all_retrieved_docs = []
        for q in query_list:
            logger.info(f"Retrieving for sub-query: {q}")
            retrieved_docs = self.retrieve(q, top_k, score_threshold)
            all_retrieved_docs.extend(retrieved_docs)

        # De-duplicate and sort all retrieved documents by score
        unique_docs_map = {doc['id']: doc for doc in reversed(all_retrieved_docs)}
        unique_docs_list = list(unique_docs_map.values())

        # Perform a global rerank against the original query to find the best documents overall.
        # This is crucial because scores from different sub-queries are not comparable.
        reranked_docs = self._maybe_rerank(query, unique_docs_list)
        return reranked_docs

    def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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
