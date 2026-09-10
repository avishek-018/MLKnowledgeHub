"""Utilities for splitting ingested text into chunks."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[TextChunk]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Full document text.
        document_id: Unique document identifier.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of TextChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk_text_value = text[start:end].strip()

        if chunk_text_value:
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    document_id=document_id,
                    text=chunk_text_value,
                    start_char=start,
                    end_char=end,
                )
            )

        if end == len(text):
            break

        start = end - overlap
        chunk_index += 1

    return chunks