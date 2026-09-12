"""General parsing utilities for ingestion workflows."""

from pathlib import Path

from ml_knowledge_hub.ingestion.pdf_parser import (
    extract_text_from_pdf,
)
from ml_knowledge_hub.ingestion.text_parsers import (
    extract_text_from_json,
    extract_text_from_markdown,
)


def extract_text(
    file_path: str | Path,
) -> str:
    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    if suffix in {".md", ".markdown"}:
        return extract_text_from_markdown(file_path)

    if suffix == ".json":
        return extract_text_from_json(file_path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )