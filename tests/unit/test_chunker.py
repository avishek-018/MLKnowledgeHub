"""Tests for text chunking utilities."""

import pytest

from ml_knowledge_hub.ingestion.chunker import chunk_text


def test_chunk_text_creates_chunks():
    text = "A" * 2500

    chunks = chunk_text(
        text=text,
        document_id="paper_001",
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) > 1
    assert chunks[0].document_id == "paper_001"
    assert chunks[0].chunk_id == "paper_001_chunk_0000"


def test_empty_text_returns_empty_list():
    chunks = chunk_text(
        text="",
        document_id="paper_001",
    )

    assert chunks == []


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            text="hello world",
            document_id="paper_001",
            chunk_size=100,
            overlap=100,
        )