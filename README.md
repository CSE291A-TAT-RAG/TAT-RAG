# TAT-RAG

A production-ready Retrieval-Augmented Generation stack that pairs **Qdrant** vector search with configurable **local or cloud LLMs**. The project ships with a Streamlit UI, CLI utilities, and RAGAS-based evaluation to help you move from ingestion to quality monitoring quickly.

## Highlights
- **Flexible LLM backends** - run locally through Ollama or swap to AWS Bedrock/Gemini via `.env`.
- **Table-aware ingestion** - pre-chunked JSONL + CSV archives hydrate tabular context before embedding.
- **One-click Streamlit UI** - chat with your data, inspect retrieved sources, and open originals.
- **CLI workflow** - ingest, query, and evaluate from `main.py`.
- **RAGAS evaluation** - measure answer quality with reproducible cached responses.
- **Docker-first setup** - all services orchestrated via `docker-compose`, GPU passthrough supported.

## Prerequisites
- Docker & Docker Compose
- Ollama running on the host (for local inference) or credentials for your chosen cloud provider
- Optional: NVIDIA GPU + drivers for accelerated embeddings/LLM

## Quick Start
```bash
# 1. Configure environment
cp .env.example .env
# edit .env to point at your Ollama/GPU settings as needed
# optional: enable hybrid retrieval by setting HYBRID_SEARCH_ENABLED=true

# 2. Launch services
docker-compose up -d

# (Optional) Reset Qdrant collection before re-ingesting
docker-compose exec rag-app python -c "from qdrant_client import QdrantClient; QdrantClient(host='qdrant', port=6333).delete_collection('documents')"

# 3. Ingest pre-chunked data (JSONL + CSV packed in a zip)
docker-compose exec rag-app python main.py ingest /app/data/chunks_all.zip

# 4. Run a query from the CLI
docker-compose exec rag-app python main.py query \
  --top-k 5 \
  "What is the A10 Networks' total cost of revenue in 2019?"

# 5. (Optional) Open the Streamlit chat UI
docker-compose exec rag-app streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# visit http://localhost:8501
```

## CLI Reference
```bash
# Retrieve only (no generation)
docker-compose exec rag-app python main.py retrieve "Explain the revenue trend."

# Query with retrieval + generation
docker-compose exec rag-app python main.py query --top-k 3 "Summarise Item 7 risk factors."

# Evaluate with a golden set
docker-compose exec rag-app python main.py evaluate \
  --json-path /app/golden_set/golden_set.json \
  --json-question-key query \
  --json-answer-key answer \
  --output /app/output/report.txt
```

## Configuration
Update `.env` to control the pipeline:

- `LLM_PROVIDER`, `LLM_MODEL`, `OLLAMA_BASE_URL`: select the generator model. Increase `LLM_MAX_TOKENS` when using thinking-capable models.
- `GEMINI_API_KEY`, `LLM_REQUEST_INTERVAL`: required when `LLM_PROVIDER=gemini`. Set a small interval (e.g., 6s) to avoid API throttling during evaluation.
- `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`: retrieval fan-out and similarity gating.
- `HYBRID_SEARCH_ENABLED`: set to `true` to fuse vector search with BM25 keyword search via Qdrant's Fusion API (requires re-ingestion after toggling).
- `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_CACHE_DIR`: embedding backend configuration.
- `QDRANT_HOST`, `QDRANT_COLLECTION`: vector store parameters (overridden inside Docker to `qdrant`).

Restart the `rag-app` service after changing environment variables:
```bash
docker-compose restart rag-app
```

## Project Structure
```
TAT-RAG/
|-- app.py                 # Streamlit UI for chat + source inspection
|-- docker-compose.yml     # Orchestrates Qdrant and rag-app containers
|-- Dockerfile             # Builds the Python runtime used by rag-app
|-- main.py                # CLI entry point (ingest / query / evaluate)
|-- requirements.txt       # Python dependencies
|-- .env                   # Runtime configuration (gitignored)
|-- data/                  # Sample ingestion archives (mounted into container)
|-- golden_set/            # Evaluation datasets
|-- models/                # Local embedding/LLM cache (mounted)
|-- output/                # Generated evaluation and cache artifacts
|-- scripts/
|   `-- precompute_answers.py  # Cache LLM outputs for RAGAS runs
`-- src/
    |-- __init__.py
    |-- config.py              # Dataclass-driven configuration loader
    |-- embedding_providers.py # Embedding abstraction (local BGE)
    |-- evaluation.py          # RAGAS evaluator and reporting helpers
    |-- ingestion.py           # Loads pre-chunked archives into Qdrant
    |-- llm_providers.py       # Ollama / Bedrock / Gemini adapters
    |-- retrieval.py           # Core RAG pipeline (retrieve + generate)
    `-- parsers/               # Legacy PDF parsers kept for reference
```

## Evaluating with RAGAS
1. **Pre-compute responses** to avoid repeated LLM calls (run inside the container via `docker-compose exec`):
   ```bash
   docker-compose exec rag-app python scripts/precompute_answers.py \
     --input /app/golden_set/golden_set.json \
     --question-field query \
     --ground-truth-field answer \
     --include-docs \
     --output /app/output/cached_answers.json
   ```
2. **Score the cached answers**:
   ```bash
   docker-compose exec rag-app python main.py evaluate \
     --cache-path /app/output/cached_answers.json \
     --output /app/output/ragas_report.txt
   ```
The resulting report summarises faithfulness, context precision/recall, answer relevancy, and correctness.


## Docker Services
| Service   | Purpose                          | Ports        |
|-----------|----------------------------------|--------------|
| `qdrant`  | Vector database + web UI         | 6333, 6334   |
| `rag-app` | Python RAG pipeline + Streamlit  | 8501 (UI)    |

GPU passthrough is pre-configured in `docker-compose.yml`. Comment the `deploy.resources` block if you do not have an NVIDIA GPU.

## Tech Stack
- **Vector DB:** Qdrant (cosine similarity)
- **Embeddings:** BAAI BGE-M3 (SentenceTransformers)
- **LLMs:** Ollama (qwen3 8b) with optional Bedrock/Gemini
- **Evaluation:** RAGAS(gemini-2.5-flash-lite)
- **Frontend:** Streamlit

## Acknowledgements
- [Ollama](https://ollama.com/)
- [Qdrant](https://qdrant.tech/)
- [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [RAGAS](https://github.com/explodinggradients/ragas)
