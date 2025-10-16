#!/bin/bash
# Setup script for Ollama model

echo "=========================================="
echo "TAT-RAG Ollama Setup"
echo "=========================================="
echo ""

# Check if ollama container is running
if ! docker ps | grep -q ollama; then
    echo "Error: Ollama container is not running"
    echo "Please start it with: docker-compose up -d ollama"
    exit 1
fi

echo "Pulling qwen3:8b model..."
echo "This may take several minutes (model size: ~5GB)"
echo ""

docker exec -it ollama ollama pull qwen3:8b

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Model qwen2.5:8b is ready to use."
    echo ""
    echo "You can now:"
    echo "1. Run the RAG pipeline: docker-compose up -d"
    echo "2. Test with examples: docker-compose exec rag-app python examples/example_usage.py"
    echo ""
else
    echo ""
    echo "Error: Failed to pull model"
    exit 1
fi
