"""PyMuPDF (fitz) based PDF parser for high-quality text extraction."""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for FitzParser. "
        "Install it with: pip install pymupdf"
    )

from .base import BaseParser

logger = logging.getLogger(__name__)


class FitzParser(BaseParser):
    """
    Advanced PDF parser using PyMuPDF (fitz).

    Features:
    - Respects PDF text block structure
    - Sorts blocks in reading order (top-to-bottom, left-to-right)
    - Normalizes text (removes excessive whitespace)
    - Handles encrypted PDFs
    - Splits blocks into paragraphs by blank lines
    """

    def __init__(self, min_paragraph_length: int = 3):
        """
        Initialize FitzParser.

        Args:
            min_paragraph_length: Minimum character length for valid paragraphs
        """
        self.min_paragraph_length = min_paragraph_length

    def parse(self, path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Parse PDF file(s) using PyMuPDF.

        Args:
            path: Path to PDF file or directory
            **kwargs: Additional options (unused for now)

        Returns:
            List of document dictionaries with content and metadata
        """
        path_obj = self.validate_path(path)
        documents = []

        if path_obj.is_file():
            if path_obj.suffix.lower() == '.pdf':
                documents.extend(self._parse_single_pdf(path_obj))
            else:
                raise ValueError(f"FitzParser only supports PDF files, got: {path_obj.suffix}")
        elif path_obj.is_dir():
            pdf_files = sorted(path_obj.rglob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in directory: {path}")
                return []

            for pdf_file in pdf_files:
                try:
                    documents.extend(self._parse_single_pdf(pdf_file))
                except Exception as e:
                    logger.error(f"Failed to parse {pdf_file}: {e}")
                    continue

        logger.info(f"FitzParser extracted {len(documents)} paragraphs from {path}")
        return documents

    def _parse_single_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of document dictionaries
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {pdf_path} ({e})")
            return []

        # Handle encrypted PDFs
        if doc.is_encrypted:
            try:
                doc.authenticate("")
            except Exception:
                logger.warning(f"Skipping encrypted PDF (cannot open): {pdf_path}")
                doc.close()
                return []

        documents = []

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)

            try:
                # Get text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
            except Exception as e:
                logger.warning(f"get_text('blocks') failed on {pdf_path} page {page_idx+1}: {e}")
                continue

            # Sort blocks in reading order (top to bottom, left to right)
            blocks = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))

            paragraphs = []
            block_positions = []  # Store position info for each paragraph

            for block_idx, block in enumerate(blocks):
                if len(block) < 7:  # Ensure we have all block components
                    continue

                # Extract block coordinates and metadata
                x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
                raw_text = block[4] or ""
                block_no = block[5]
                block_type = block[6]

                normalized_text = self._normalize_block_text(raw_text)

                if not normalized_text:
                    continue

                # Split block into paragraphs by blank lines
                paras = self._split_into_paragraphs(normalized_text)

                # Store position info for each paragraph from this block
                for para in paras:
                    paragraphs.append(para)
                    block_positions.append({
                        "block_no": block_no,
                        "block_type": block_type,
                        "bbox": {
                            "x0": round(x0, 2),
                            "y0": round(y0, 2),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2)
                        }
                    })

            # Create document entries for each paragraph
            for para_idx, (para, position) in enumerate(zip(paragraphs, block_positions), start=1):
                if len(para.strip()) < self.min_paragraph_length:
                    continue

                documents.append({
                    "content": para,
                    "metadata": {
                        "source": str(pdf_path),
                        "page": page_idx + 1,  # 1-based page numbering
                        "para_index": para_idx,
                        "char_count": len(para),
                        "parser": "fitz",
                        # Original document position info
                        "block_no": position["block_no"],
                        "block_type": position["block_type"],
                        "bbox": position["bbox"]
                    }
                })

        doc.close()
        return documents

    @staticmethod
    def _normalize_block_text(text: str) -> str:
        """
        Normalize raw block text.

        - Convert CR to LF
        - Trim trailing spaces on each line
        - Collapse 3+ newlines to 2 (keep paragraph boundaries)
        - Strip outer whitespace

        Args:
            text: Raw text from PDF block

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert carriage returns to newlines
        text = text.replace("\r", "\n")

        # Strip trailing spaces on each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Collapse excessive blank lines (max 2 newlines = 1 blank line)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split block text into paragraphs by blank lines.

        Args:
            text: Normalized block text

        Returns:
            List of paragraph strings
        """
        if not text:
            return []

        # Split on one or more blank lines
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean and filter
        cleaned = [
            p.strip()
            for p in paragraphs
            if p and p.strip() and len(p.strip()) >= self.min_paragraph_length
        ]

        return cleaned
