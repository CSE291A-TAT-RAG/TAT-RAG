"""Main entry point for the RAG pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import RAGConfig
from src.ingestion import DocumentIngestion
from src.retrieval import RAGPipeline
from src.evaluation import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_command(args):
    """Handle document ingestion command."""
    config = RAGConfig()
    ingestion = DocumentIngestion(config)

    overwrite = not args.no_overwrite  # Convert --no-overwrite flag to overwrite bool
    count = ingestion.ingest(
        args.path,
        args.file_type,
        parser_type=args.parser,
        overwrite=overwrite
    )
    logger.info(f"Successfully ingested {count} chunks using {args.parser} parser")


def retrieve_command(args):
    """Handle retrieve-only command (no LLM generation)."""
    config = RAGConfig()
    rag = RAGPipeline(config)

    retrieved_docs = rag.retrieve(args.query, top_k=args.top_k)

    print("\n" + "="*50)
    print("QUERY:", args.query)
    print("="*50)
    print(f"\nRetrieved {len(retrieved_docs)} documents:\n")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"Document {i} (Score: {doc['score']:.4f}):")
        print("-" * 50)
        print(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""))
        print()


def query_command(args):
    """Handle query command (retrieve + generation)."""
    config = RAGConfig()
    rag = RAGPipeline(config)

    result = rag.query(args.query, top_k=args.top_k)

    print("\n" + "="*50)
    print("QUERY:", result["query"])
    print("="*50)
    print("\nANSWER:")
    print(result["answer"])
    print("\n" + "="*50)
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    print(f"Model: {result['model']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")


def evaluate_command(args):
    """Handle evaluation command."""
    config = RAGConfig()
    rag = RAGPipeline(config)
    evaluator = RAGEvaluator(rag)

    if args.question and args.ground_truth:
        result = evaluator.evaluate(
            questions=[args.question],
            ground_truths=[args.ground_truth]
        )
    elif args.csv_path:
        result = evaluator.evaluate_from_file(
            args.csv_path,
            question_col=args.question_col,
            ground_truth_col=args.ground_truth_col
        )
    else:
        logger.error("Evaluation requires either --csv-path or both --question and --ground-truth")
        sys.exit(1)

    # Generate and display report
    report = evaluator.generate_report(result, args.output)
    print(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="TAT-RAG: RAG Pipeline with Qdrant and RAGAS")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Qdrant")
    ingest_parser.add_argument("path", help="Path to file or directory")
    ingest_parser.add_argument("--file-type", default="txt", choices=["txt", "pdf"],
                               help="Type of files to ingest")
    ingest_parser.add_argument("--parser", default="langchain", choices=["langchain", "fitz"],
                               help="Parser to use: 'langchain' (simple, fast) or 'fitz' (advanced PDF parsing)")
    ingest_parser.add_argument("--no-overwrite", action="store_true",
                               help="Append new chunks without deleting existing ones from the same source")
    ingest_parser.set_defaults(func=ingest_command)

    # Retrieve command (no LLM generation)
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant documents (no generation)")
    retrieve_parser.add_argument("query", help="Query string")
    retrieve_parser.add_argument("--top-k", type=int, default=5,
                                help="Number of documents to retrieve")
    retrieve_parser.set_defaults(func=retrieve_command)

    # Query command (retrieve + generate)
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline (retrieve + generate)")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--top-k", type=int, default=5,
                             help="Number of documents to retrieve")
    query_parser.set_defaults(func=query_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate RAG pipeline with RAGAS")
    eval_parser.add_argument("--csv-path",
                            help="Path to CSV file with questions and ground truths")
    eval_parser.add_argument("--question", help="Single question to evaluate directly")
    eval_parser.add_argument("--ground-truth", help="Ground truth for the single question")
    eval_parser.add_argument("--question-col", default="question",
                            help="Name of question column")
    eval_parser.add_argument("--ground-truth-col", default="ground_truth",
                            help="Name of ground truth column")
    eval_parser.add_argument("--output", help="Path to save evaluation report")
    eval_parser.set_defaults(func=evaluate_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
