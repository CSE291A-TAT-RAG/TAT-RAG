.PHONY: help build up down logs ingest query evaluate clean test

help:
	@echo "TAT-RAG Makefile Commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs"
	@echo "  make ingest      - Ingest sample documents"
	@echo "  make query       - Run sample query"
	@echo "  make evaluate    - Run evaluation"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make test        - Run tests"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started. Waiting for Qdrant to be ready..."
	@sleep 5
	@echo "Services are ready!"

down:
	docker-compose down

logs:
	docker-compose logs -f

ingest:
	docker-compose exec rag-app python main.py ingest /app/data/sample.txt --file-type txt

query:
	docker-compose exec rag-app python main.py query "What is RAG?" --top-k 5

evaluate:
	docker-compose exec rag-app python main.py evaluate \
		--csv-path /app/examples/eval_dataset_example.csv \
		--question-col question \
		--ground-truth-col ground_truth

clean:
	docker-compose down -v
	rm -rf qdrant_storage/
	rm -rf data/*.txt
	rm -rf output/*.txt

test:
	docker-compose exec rag-app python -m pytest tests/ -v

shell:
	docker-compose exec rag-app bash

install-local:
	pip install -r requirements.txt

run-local-ingest:
	python main.py ingest data/sample.txt --file-type txt

run-local-query:
	python main.py query "What is RAG?" --top-k 5
