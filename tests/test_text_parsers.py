"""Tests for text parsing utilities."""

import json

import pytest

from ml_knowledge_hub.ingestion.text_parsers import (
    extract_text_from_json,
    extract_text_from_markdown,
)


def test_markdown_parser(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(
        "# Model Card\nThis is a model.",
        encoding="utf-8",
    )

    text = extract_text_from_markdown(path)

    assert "Model Card" in text
    assert "This is a model" in text


def test_json_parser(tmp_path):
    path = tmp_path / "sample.json"

    path.write_text(
        json.dumps(
            {
                "model": "ResNet50",
                "accuracy": 0.91,
            }
        ),
        encoding="utf-8",
    )

    text = extract_text_from_json(path)

    assert "ResNet50" in text
    assert "accuracy" in text


def test_markdown_parser_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text_from_markdown(
            "missing.md"
        )