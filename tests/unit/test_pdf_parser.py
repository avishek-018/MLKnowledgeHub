"""Tests for PDF ingestion utilities."""

from pathlib import Path

import pytest

from ml_knowledge_hub.ingestion.pdf_parser import extract_text_from_pdf


def test_pdf_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("does_not_exist.pdf")


def test_reject_non_pdf(tmp_path: Path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello")

    with pytest.raises(ValueError):
        extract_text_from_pdf(file_path)