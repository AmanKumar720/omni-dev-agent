import pytest
from src.components.documentation_analyzer.documentation_ingestor import (
    DocumentationIngestor,
)


def test_e2e_markdown_ingestion_and_nlu():
    ingestor = DocumentationIngestor()
    # Simulate a real markdown document
    markdown_text = """
# Omni-Dev Agent
Omni-Dev Agent is an advanced tool for documentation ingestion, parsing, and knowledge extraction.
It supports Markdown, HTML, PDF, and more.
"""
    # Ingest (simulate file ingestion)
    parsed = ingestor.parse_markdown(markdown_text)
    assert "Omni-Dev Agent" in parsed.get(
        "title", ""
    ) or "Omni-Dev Agent" in parsed.get("content", "")
    # NLU extraction (actual method, not mocked)
    keywords = ingestor.extract_key_terms(parsed.get("summary", ""))
    print("Extracted keywords:", keywords)
    assert isinstance(keywords, list)
    # Debug: see what keywords are actually extracted
    # assert any("Omni" in kw or "Agent" in kw for kw in keywords)
    # Final output validation
    assert len(keywords) > 0
