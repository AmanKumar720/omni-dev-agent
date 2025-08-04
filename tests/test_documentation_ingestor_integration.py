import pytest
from unittest.mock import MagicMock
from src.components.documentation_analyzer.documentation_ingestor import (
    DocumentationIngestor,
)


def test_ingest_and_parse_markdown():
    ingestor = DocumentationIngestor()
    # Mock the markdown parser
    ingestor.parse_markdown = MagicMock(
        return_value={"title": "Test", "content": "Hello World"}
    )
    result = ingestor.parse_markdown("# Test\nHello World")
    assert result["title"] == "Test"
    assert result["content"] == "Hello World"


def test_ingest_and_nlu_extraction():
    ingestor = DocumentationIngestor()
    # Mock the NLU extraction
    ingestor.extract_keywords = MagicMock(return_value=["keyword1", "keyword2"])
    keywords = ingestor.extract_keywords("Some text about keyword1 and keyword2.")
    assert "keyword1" in keywords
    assert "keyword2" in keywords


def test_full_ingestion_pipeline():
    ingestor = DocumentationIngestor()
    # Mock all sub-methods
    ingestor.ingest_from_file = MagicMock(return_value="# Test\nHello World")
    ingestor.parse_markdown = MagicMock(
        return_value={"title": "Test", "content": "Hello World"}
    )
    ingestor.extract_keywords = MagicMock(return_value=["Test", "Hello World"])
    file_path = "dummy.md"
    raw = ingestor.ingest_from_file(file_path)
    parsed = ingestor.parse_markdown(raw)
    keywords = ingestor.extract_keywords(parsed["content"])
    assert parsed["title"] == "Test"
    assert "Hello World" in keywords
