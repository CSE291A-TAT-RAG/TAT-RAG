#!/usr/bin/env python3
"""
Quick preview of parsed PDF - shows first few paragraphs.

Usage:
    python scripts/preview_pdf.py <pdf_path> [--parser fitz] [--num 5]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import create_parser


def main():
    parser = argparse.ArgumentParser(description="Quick preview of parsed PDF")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--parser", default="fitz", choices=["langchain", "fitz"],
                       help="Parser to use (default: fitz)")
    parser.add_argument("--num", type=int, default=5,
                       help="Number of paragraphs to show (default: 5)")

    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"❌ Error: File not found: {args.pdf_path}")
        sys.exit(1)

    print(f"📄 Parsing: {args.pdf_path}")
    print(f"🔧 Parser: {args.parser}\n")

    # Parse PDF
    pdf_parser = create_parser(args.parser)
    documents = pdf_parser.parse(args.pdf_path, file_type="pdf")

    print(f"{'='*70}")
    print(f"Total paragraphs extracted: {len(documents)}")
    print(f"{'='*70}\n")

    # Show first N paragraphs
    for i, doc in enumerate(documents[:args.num], 1):
        metadata = doc['metadata']
        content = doc['content']

        # Header
        print(f"┌─ Paragraph {i}/{len(documents)} " + "─" * 50)
        print(f"│ Page: {metadata.get('page', 'N/A')}")
        print(f"│ Length: {len(content)} characters")
        if 'para_index' in metadata:
            print(f"│ Paragraph index: {metadata['para_index']}")
        print(f"└" + "─" * 68)

        # Content
        print(content)
        print("\n" + "─" * 70 + "\n")

    # Summary
    if len(documents) > args.num:
        print(f"... and {len(documents) - args.num} more paragraphs")
        print(f"\n💡 Use --num {len(documents)} to see all paragraphs")


if __name__ == "__main__":
    main()
