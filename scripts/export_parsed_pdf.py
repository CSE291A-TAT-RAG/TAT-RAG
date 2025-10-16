#!/usr/bin/env python3
"""
Export parsed PDF results to JSON/JSONL for inspection.

Usage:
    python scripts/export_parsed_pdf.py <pdf_path> [--parser fitz] [--output results.jsonl]
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import create_parser


def export_to_jsonl(documents, output_path):
    """Export documents to JSONL format (one JSON per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documents, 1):
            record = {
                "index": i,
                "content": doc["content"],
                "metadata": doc["metadata"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ Exported {len(documents)} documents to {output_path}")


def export_to_json(documents, output_path):
    """Export documents to pretty-printed JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported {len(documents)} documents to {output_path}")


def print_summary(documents):
    """Print summary statistics."""
    total_chars = sum(len(doc['content']) for doc in documents)
    avg_chars = total_chars / len(documents) if documents else 0

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total documents: {len(documents)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chars per document: {avg_chars:.0f}")
    if documents:
        print(f"Min chars: {min(len(doc['content']) for doc in documents)}")
        print(f"Max chars: {max(len(doc['content']) for doc in documents)}")

    # Page distribution
    if documents and 'page' in documents[0]['metadata']:
        pages = [doc['metadata'].get('page', 0) for doc in documents]
        print(f"\nPage range: {min(pages)} - {max(pages)}")
        print(f"Total pages: {max(pages)}")


def print_samples(documents, num_samples=3):
    """Print sample documents."""
    print(f"\n{'='*60}")
    print(f"Sample Documents (first {min(num_samples, len(documents))})")
    print(f"{'='*60}")

    for i, doc in enumerate(documents[:num_samples], 1):
        print(f"\n--- Document {i} ---")
        print(f"Metadata: {doc['metadata']}")
        print(f"Content ({len(doc['content'])} chars):")
        preview = doc['content'][:300]
        print(f"{preview}...")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export parsed PDF to JSON/JSONL")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--parser", default="fitz", choices=["langchain", "fitz"],
                       help="Parser to use (default: fitz)")
    parser.add_argument("--output", help="Output file path (*.json or *.jsonl)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl",
                       help="Output format (default: jsonl)")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of sample documents to display (default: 3)")
    parser.add_argument("--no-preview", action="store_true",
                       help="Don't print preview to console")

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.pdf_path).exists():
        print(f"❌ Error: File not found: {args.pdf_path}")
        sys.exit(1)

    print(f"📄 Parsing PDF: {args.pdf_path}")
    print(f"🔧 Using parser: {args.parser}")

    # Parse PDF
    pdf_parser = create_parser(args.parser)
    documents = pdf_parser.parse(args.pdf_path, file_type="pdf")

    # Print summary
    print_summary(documents)

    # Print samples
    if not args.no_preview:
        print_samples(documents, args.samples)

    # Export to file
    if args.output:
        output_path = args.output
        # Auto-detect format from extension
        if output_path.endswith('.json'):
            export_to_json(documents, output_path)
        elif output_path.endswith('.jsonl'):
            export_to_jsonl(documents, output_path)
        else:
            # Use specified format
            if args.format == 'jsonl':
                export_to_jsonl(documents, output_path)
            else:
                export_to_json(documents, output_path)
    else:
        # Auto-generate filename
        pdf_name = Path(args.pdf_path).stem
        output_path = f"{pdf_name}_{args.parser}_parsed.jsonl"
        export_to_jsonl(documents, output_path)

    print(f"\n💡 Tip: View JSONL with:")
    print(f"   cat {output_path} | jq .")
    print(f"   or open with any text editor")


if __name__ == "__main__":
    main()
