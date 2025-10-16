"""LangChain-based parser for simple document loading (TXT, PDF)."""

import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)

from .base import BaseParser

logger = logging.getLogger(__name__)


class LangChainParser(BaseParser):
    """
    Simple document parser using LangChain loaders.

    Features:
    - Fast and simple loading
    - Supports TXT and PDF files
    - Directory batch processing
    - Uses PyPDFLoader for PDFs (page-level extraction)

    Note: For more advanced PDF parsing, use FitzParser instead.
    """

    def __init__(self):
        """Initialize LangChainParser."""
        pass

    def parse(self, path: str, file_type: str = "txt", **kwargs) -> List[Dict[str, Any]]:
        """
        Parse document(s) using LangChain loaders.

        Args:
            path: Path to file or directory
            file_type: Type of files to load ("txt" or "pdf")
            **kwargs: Additional options (unused)

        Returns:
            List of document dictionaries with content and metadata
        """
        path_obj = self.validate_path(path)
        documents = []

        if path_obj.is_file():
            documents = self._load_file(path_obj, file_type)
        elif path_obj.is_dir():
            documents = self._load_directory(path_obj, file_type)
        else:
            raise ValueError(f"Invalid path: {path}")

        logger.info(f"LangChainParser loaded {len(documents)} documents from {path}")
        return documents

    def _load_file(self, path: Path, file_type: str) -> List[Dict[str, Any]]:
        """
        Load a single file.

        Args:
            path: Path to the file
            file_type: Type of file ("txt" or "pdf")

        Returns:
            List of document dictionaries
        """
        if file_type == "pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))

        try:
            docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            return []

        return self._convert_to_standard_format(docs)

    def _load_directory(self, path: Path, file_type: str) -> List[Dict[str, Any]]:
        """
        Load all files from a directory.

        Args:
            path: Path to the directory
            file_type: Type of files to load ("txt" or "pdf")

        Returns:
            List of document dictionaries
        """
        if file_type == "pdf":
            loader = DirectoryLoader(
                str(path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
        else:
            loader = DirectoryLoader(
                str(path),
                glob="**/*.txt",
                loader_cls=TextLoader
            )

        try:
            docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to load directory {path}: {e}")
            return []

        return self._convert_to_standard_format(docs)

    def _convert_to_standard_format(self, langchain_docs) -> List[Dict[str, Any]]:
        """
        Convert LangChain documents to standard format.

        Args:
            langchain_docs: List of LangChain Document objects

        Returns:
            List of standardized document dictionaries
        """
        documents = []
        for doc in langchain_docs:
            documents.append({
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "parser": "langchain"
                }
            })
        return documents
