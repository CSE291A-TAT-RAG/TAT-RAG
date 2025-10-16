"""Base parser interface for document processing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    All parsers must implement the parse() method which returns
    a standardized list of document dictionaries.
    """

    @abstractmethod
    def parse(self, path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Parse a document and return structured content.

        Args:
            path: Path to the file or directory to parse
            **kwargs: Additional parser-specific options

        Returns:
            List of document dictionaries, each containing:
            - content: str - The text content
            - metadata: dict - Metadata about the document
              - source: str - Source file path
              - page: int (optional) - Page number for PDFs
              - para_index: int (optional) - Paragraph index
              - ... (other parser-specific metadata)

        Raises:
            FileNotFoundError: If the path doesn't exist
            ValueError: If the file format is not supported
        """
        pass

    def validate_path(self, path: str) -> Path:
        """
        Validate that the path exists.

        Args:
            path: Path to validate

        Returns:
            Path object

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path_obj
