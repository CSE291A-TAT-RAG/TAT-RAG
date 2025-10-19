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
    - Extracts full page text in reading order
    - Normalizes text (removes excessive whitespace)
    - Handles encrypted PDFs
    - Optimized for downstream chunking with RecursiveCharacterTextSplitter

    Note: This parser extracts text at the page level, which works well
    with chunking strategies. Each page becomes a document that will be
    chunked by the ingestion pipeline.
    """

    def __init__(self, min_page_length: int = 10):
        """
        Initialize FitzParser.

        Args:
            min_page_length: Minimum character length for valid page content (default: 10)
        """
        self.min_page_length = min_page_length

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

        logger.info(f"FitzParser extracted {len(documents)} pages from {path}")
        return documents

    def _parse_single_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a single PDF file, extracting full page text.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of document dictionaries (one per page)
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
                # Extract full page text
                page_text = page.get_text("text")
                page_text = self._normalize_text(page_text)

                # Only add pages with meaningful content
                if page_text and len(page_text.strip()) > self.min_page_length:
                    documents.append({
                        "content": page_text,
                        "metadata": {
                            "source": str(pdf_path),
                            "page": page_idx + 1,  # 1-based page numbering
                            "char_count": len(page_text),
                            "parser": "fitz"
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to extract text from {pdf_path} page {page_idx+1}: {e}")
                continue

        doc.close()
        return documents

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize raw PDF text.

        - Convert CR to LF
        - Trim trailing spaces on each line
        - Collapse 3+ newlines to 2 (keep paragraph boundaries)
        - Strip outer whitespace

        Args:
            text: Raw text from PDF page

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
