"""Document parsers for different file types and parsing strategies."""

from .base import BaseParser
from .langchain_parser import LangChainParser
from .fitz_parser import FitzParser

__all__ = ["BaseParser", "LangChainParser", "FitzParser", "create_parser"]


def create_parser(parser_type: str = "langchain", **kwargs) -> BaseParser:
    """
    Factory function to create parser instances.

    Args:
        parser_type: Type of parser to create ("langchain" or "fitz")
        **kwargs: Additional arguments to pass to the parser

    Returns:
        Instance of the requested parser

    Raises:
        ValueError: If parser_type is not supported
    """
    if parser_type == "fitz":
        return FitzParser(**kwargs)
    elif parser_type == "langchain":
        return LangChainParser(**kwargs)
    else:
        raise ValueError(
            f"Unknown parser type: {parser_type}. "
            f"Supported types: 'langchain', 'fitz'"
        )
