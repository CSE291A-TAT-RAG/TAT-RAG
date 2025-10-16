"""Unit tests for document parsers."""

import pytest
from pathlib import Path
from src.parsers import create_parser, BaseParser, LangChainParser, FitzParser


def test_create_parser_langchain():
    """Test creating LangChain parser."""
    parser = create_parser("langchain")
    assert isinstance(parser, LangChainParser)
    assert isinstance(parser, BaseParser)


def test_create_parser_fitz():
    """Test creating Fitz parser."""
    parser = create_parser("fitz")
    assert isinstance(parser, FitzParser)
    assert isinstance(parser, BaseParser)


def test_create_parser_invalid():
    """Test creating parser with invalid type."""
    with pytest.raises(ValueError, match="Unknown parser type"):
        create_parser("invalid_parser")


def test_fitz_parser_initialization():
    """Test Fitz parser initialization."""
    parser = FitzParser(min_paragraph_length=5)
    assert parser.min_paragraph_length == 5


def test_langchain_parser_initialization():
    """Test LangChain parser initialization."""
    parser = LangChainParser()
    assert parser is not None


def test_fitz_normalize_text():
    """Test Fitz text normalization."""
    parser = FitzParser()

    # Test carriage return conversion
    # Note: \r\n becomes \n\n because \r is replaced with \n first
    text = "Hello\r\nWorld\r\n"
    normalized = parser._normalize_block_text(text)
    assert "\r" not in normalized
    assert normalized == "Hello\n\nWorld"

    # Test excessive newlines
    text = "Line1\n\n\n\n\nLine2"
    normalized = parser._normalize_block_text(text)
    assert normalized == "Line1\n\nLine2"

    # Test trailing spaces
    text = "Line1   \nLine2  "
    normalized = parser._normalize_block_text(text)
    assert normalized == "Line1\nLine2"


def test_fitz_split_paragraphs():
    """Test Fitz paragraph splitting."""
    parser = FitzParser(min_paragraph_length=3)

    # Test basic split
    text = "Para1\n\nPara2\n\nPara3"
    paras = parser._split_into_paragraphs(text)
    assert len(paras) == 3
    assert paras == ["Para1", "Para2", "Para3"]

    # Test minimum length filtering
    # "OK" has length 2, "X" has length 1, both are filtered out (min_paragraph_length=3)
    text = "OK\n\nX\n\nGood"
    paras = parser._split_into_paragraphs(text)
    assert len(paras) == 1
    assert paras == ["Good"]
    assert "X" not in paras  # Too short
    assert "OK" not in paras  # Too short


def test_parser_validate_path_invalid():
    """Test path validation with invalid path."""
    parser = LangChainParser()
    with pytest.raises(FileNotFoundError):
        parser.validate_path("/nonexistent/path/file.pdf")


def test_fitz_parser_unsupported_file_type():
    """Test Fitz parser with non-PDF file."""
    parser = FitzParser()

    # Create a temporary txt file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="only supports PDF files"):
            parser.parse(temp_path)
    finally:
        Path(temp_path).unlink()
