# TAT-RAG 🚀

A complete, production-ready RAG (Retrieval-Augmented Generation) pipeline with **local and cloud LLM support**, Qdrant vector database, and RAGAS evaluation.

## ✨ Key Features

- **🌐 Streamlit Web UI**: Interactive chat interface with PDF source navigation
- **🔄 Flexible LLM Providers**: Ollama (local) or AWS Bedrock (cloud)
- **🎯 Retrieve-Only Mode**: Perfect for Mac/CPU-only users - no LLM required!
- **📚 Document Ingestion**: Support for TXT and PDF files
- **📄 Advanced PDF Parsing**: Dual parser support with position tracking
  - **LangChain**: Fast, simple loading for general documents
  - **Fitz (PyMuPDF)**: Advanced parsing with bbox coordinates for source navigation
- **🔍 Semantic Search**: BGE-M3 multilingual embeddings (local)
- **🤖 RAG Pipeline**: Context-aware answer generation
- **📊 RAGAS Evaluation**: Comprehensive quality metrics
- **🐳 Docker Ready**: Fully containerized with optional GPU support

## 🏗️ Architecture

The project operates in three main pipelines:

**1. Ingestion Pipeline**
```
[Source Docs] -> [ingestion.py] -> [embedding_providers.py] -> [Vector DB]
(.txt, .pdf)    (Chunking)       (Create Embeddings)          (Qdrant)
```

**2. RAG Pipeline**
```
[User Query] -> [retrieval.py] -> [embedding_providers.py] -> [Vector DB]
     |           (Search)         (Create Query Embedding)   (Similarity Search)
     |                                                          |
     +--------------------------------------------> [Retrieved Context]
     |                                                          |
     +--------------------------------------------> [llm_providers.py] -> [LLM]
                                                      (Context + Query)    (Ollama/Bedrock)
                                                                             |
                                                                             V
                                                                      [Final Answer]
```

**3. Evaluation Pipeline**
```
[Evaluation CSV] -> [evaluation.py] -> (Runs RAG Pipeline) -> [RAGAS Metrics]
(question, gt)                                                 (Faithfulness, etc.)
```

## 🚀 Quick Start

```bash
# Start all services
docker-compose up -d --build

# Ingest documents for windows
docker-compose exec rag-app python main.py ingest //app/data/sample.txt

# Ingest documents for macOS / Linux
docker-compose exec rag-app python main.py ingest /app/data/sample.txt

# Launch Streamlit Web UI
docker-compose exec rag-app streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# Then open: http://localhost:8501
```

## 📖 Usage Examples

### 🌐 Web UI (Streamlit)

The easiest way to use TAT-RAG is through the Streamlit web interface:

```bash
# Make sure your documents are ingested first
docker-compose exec rag-app python main.py ingest /app/data/your_file.pdf --file-type pdf --parser fitz

# Launch Streamlit
docker-compose exec rag-app streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Open in browser: http://localhost:8501
```

**Features:**
- 💬 **Chat Interface**: Ask questions and get answers with context
- 📄 **Source Display**: View retrieved documents with metadata in sidebar
- 📍 **Position Information**: See exact page number and bounding box coordinates
- 🔗 **PDF Navigation**: Click "View in PDF" to jump to the source page
- ⚙️ **Smart Retrieval**: Control max sources and similarity threshold
  - Adjust top-K (max results to return)
  - Set similarity threshold (filter low-quality results)

**Screenshot Features:**
- Left panel: Chat history and Q&A
- Right panel: Retrieved sources with scores and metadata
- Bottom: PDF viewer that opens when you click on a source

### CLI Commands

```bash
# Ingest documents
docker-compose exec rag-app python main.py ingest /app/data/docs.txt

# Ingest PDF with default (LangChain) parser
docker-compose exec rag-app python main.py ingest /app/data/paper.pdf --file-type pdf

# Ingest PDF with advanced Fitz parser (recommended for financial reports)
docker-compose exec rag-app python main.py ingest /app/data/financial_report.pdf \
  --file-type pdf \
  --parser fitz


# Evaluate with single question (quick test)
# Linux/macOS
docker-compose exec rag-app python main.py evaluate \
  --question "What is RAG?" \
  --ground-truth "RAG stands for Retrieval-Augmented Generation, a technique that combines information retrieval with text generation" \
  --output /app/output/report.txt

# Windows Git Bash
export MSYS_NO_PATHCONV=1 && docker-compose exec rag-app python main.py evaluate \
  --question "What is RAG?" \
  --ground-truth "RAG stands for Retrieval-Augmented Generation, a technique that combines information retrieval with text generation" \
  --output /app/output/report.txt


# Evaluate with CSV dataset (batch evaluation)
# Linux/macOS
docker-compose exec rag-app python main.py evaluate \
  --csv-path /app/examples/eval_dataset_example.csv \
  --output /app/output/batch_report.txt

# Windows Git Bash
export MSYS_NO_PATHCONV=1 && docker-compose exec rag-app python main.py evaluate \
  --csv-path /app/examples/eval_dataset_example.csv \
  --output /app/output/batch_report.txt
```

### Preview parser
```bash
# Windows Git Bash
export MSYS_NO_PATHCONV=1
docker-compose exec rag-app python scripts/preview_pdf.py /app/data/a10-networks-inc_2019.pdf

# macOs/Linux
docker-compose exec rag-app python scripts/preview_pdf.py /app/data/a10-networks-inc_2019.pdf

# 使用 LangChain parser
docker-compose exec rag-app python scripts/preview_pdf.py /app/data/a10-networks-inc_2019.pdf --parser langchain
```

### Export to JSON/JSONL
```bash
# 导出为 JSONL（每行一个 JSON，方便处理大文件）
docker-compose exec rag-app python scripts/export_parsed_pdf.py \
  /app/data/a10-networks-inc_2019.pdf \
  --output /app/output/a10_parsed.jsonl

# 导出为 JSON（完整 JSON 数组，方便阅读）
docker-compose exec rag-app python scripts/export_parsed_pdf.py \
  /app/data/a10-networks-inc_2019.pdf \
  --output /app/output/a10_parsed.json \
  --format json

# 不显示预览，只导出
docker-compose exec rag-app python scripts/export_parsed_pdf.py \
  /app/data/a10-networks-inc_2019.pdf \
  --output /app/output/a10_parsed.jsonl \
  --no-preview
```

## E2E Sample Test

```bash
python e2e_test.py
```

## Unit Test

```bash
docker-compose exec rag-app pytest
```


## ⚙️ Configuration

### 🚀 Quick Setup (First Time)

**Step 1: Create your configuration file**
```bash
# Copy the template to create your .env file
cp .env.example .env
```

**Step 2: Choose your LLM provider** (edit `.env`)

**Option A: Local Ollama (Default)**
```bash
# Already set by default in .env, no changes needed!
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
```

**Option B: AWS Bedrock (Claude)**
```bash
# Edit .env and change these lines:
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_actual_access_key
AWS_SECRET_ACCESS_KEY=your_actual_secret_key
```

**Step 3: Start Docker**
```bash
docker-compose up -d
```

> **Note:** Your `.env` file is git-ignored for security. Never commit AWS credentials!

### Switching Between Providers

Simply edit your `.env` file and change the `LLM_PROVIDER` line, then restart Docker:

```bash
docker-compose down
docker-compose up -d
```

### Available Models

**Ollama Models:**
- `qwen3:8b`
- Any model from [Ollama library](https://ollama.com/library)

**AWS Bedrock Models:**
- `anthropic.claude-3-haiku-20240307-v1:0` (Fastest, Cheapest)

## 📊 RAGAS Evaluation Metrics

- **Faithfulness**: How grounded the answer is in retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: Quality of retrieved documents ranking
- **Context Recall**: Coverage of ground truth in retrieved context
- **Answer Correctness**: Similarity to ground truth answer

Example evaluation dataset (`examples/eval_dataset_example.csv`):
```csv
question,ground_truth
"What is RAG?","RAG combines information retrieval with text generation"
"What is Qdrant used for?","Qdrant is a vector database for similarity search"
```


## 🐳 Docker Services

```yaml
services:
  qdrant:    # Vector database (6333, 6334)
  ollama:    # Local LLM server (11434)
  rag-app:   # Your RAG application
```

### GPU Support

Uncomment in `docker-compose.yml`:
```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Then set: `EMBEDDING_DEVICE=cuda` in `.env`

## 📁 Project Structure

```
TAT-RAG/
├── src/                      # Core application logic
│   ├── config.py             # Manages all configurations from .env file
│   ├── llm_providers.py      # Interface for different LLMs (Ollama, Bedrock)
│   ├── embedding_providers.py# Interface for embedding models (local BGE)
│   ├── parsers/              # Document parsing strategies
│   │   ├── base.py           # Abstract parser interface
│   │   ├── langchain_parser.py  # Simple, fast parser
│   │   └── fitz_parser.py    # Advanced PDF parser (PyMuPDF)
│   ├── ingestion.py          # Handles document reading, chunking, and embedding storage
│   ├── retrieval.py          # Performs semantic search and answer generation
│   └── evaluation.py         # Calculates RAG quality metrics using RAGAS
├── data/                     # (Recommended) Directory for your source documents
├── examples/                 # Example files for testing and evaluation
│   ├── eval_dataset_example.csv
│   └── test_parsers.py       # Compare LangChain vs Fitz parsers
├── output/                   # (Generated) Default directory for evaluation reports
├── scripts/                  # Helper scripts
│   └── setup_ollama.sh       # Setup script for Linux/Mac
├── main.py                   # Main CLI entry point for all operations (ingest, evaluate)
├── e2e_test.py               # Automated end-to-end test script
├── docker-compose.yml        # Defines and orchestrates all services (Qdrant, Ollama, App)
├── Dockerfile                # Builds the Python application container
├── requirements.txt          # Python package dependencies
├── .env.example              # Template for environment configuration (commit this)
├── .env                      # Your actual configuration (git-ignored, NEVER commit!)
└── README.md                 # This file
```

## 🌟 Tech Stack

- **Vector Database**: Qdrant (cosine similarity)
- **LLM**: Ollama (Qwen/Llama/Mistral) or AWS Bedrock (Claude)
- **Embeddings**: BGE-M3 (local)
- **Evaluation**: RAGAS framework
- **Document Loading**: LangChain loaders + PyMuPDF
- **Deployment**: Docker & Docker Compose

## 📄 PDF Parser Selection Guide

### When to Use Which Parser?

| Parser | Best For | Pros | Cons |
|--------|----------|------|------|
| **langchain** (default) | Simple documents, TXT files, quick testing | ✅ Fast<br>✅ Simple<br>✅ Multiple formats | ❌ Basic PDF extraction<br>❌ Poor paragraph boundaries |
| **fitz** (PyMuPDF) | Financial reports, complex PDFs, production use | ✅ Respects PDF structure<br>✅ Better text quality<br>✅ Reading order sorting<br>✅ Handles encrypted PDFs | ❌ PDF only<br>❌ Slightly slower |

### Example Usage

```bash
# Test both parsers on your PDF
docker-compose exec rag-app python examples/test_parsers.py /app/data/your_file.pdf

# Use LangChain parser (default)
docker-compose exec rag-app python main.py ingest /app/data/report.pdf --file-type pdf

# Use Fitz parser (recommended for financial documents)
docker-compose exec rag-app python main.py ingest /app/data/report.pdf --file-type pdf --parser fitz
```

### Key Differences

**LangChain Parser:**
- Uses PyPDFLoader internally
- Page-level extraction
- Simple and fast
- Good for basic documents

**Fitz Parser (PyMuPDF):**
- Extracts text blocks in reading order (top→bottom, left→right)
- Preserves paragraph boundaries
- Better text normalization
- Handles edge cases (encrypted PDFs, complex layouts)
- Ideal for financial reports (10-K, 10-Q, annual reports)

## 📚 Documentation

- [examples/example_usage.py](examples/example_usage.py) - Python API examples
- [.env.example](.env.example) - All configuration options


## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation
- [BGE](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [LangChain](https://langchain.com/) - Document loaders
