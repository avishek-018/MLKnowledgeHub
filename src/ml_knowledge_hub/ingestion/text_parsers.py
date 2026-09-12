"""Text parsing utilities for ingestion workflows."""

import json
from pathlib import Path


def extract_text_from_markdown(
    file_path: str | Path,
) -> str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("Input file must be Markdown.")

    return file_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


def extract_text_from_json(
    file_path: str | Path,
) -> str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".json":
        raise ValueError("Input file must be JSON.")

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )