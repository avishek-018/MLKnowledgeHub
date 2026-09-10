"""Pipeline orchestration for ingestion workflows."""

from pathlib import Path

from ml_knowledge_hub.ingestion.pdf_parser import extract_text_from_pdf
from ml_knowledge_hub.ingestion.chunker import TextChunk, chunk_text


def ingest_pdf(
    pdf_path: str | Path,
    document_id: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    """
    Extract text from a PDF and split it into structured chunks.
    """
    text = extract_text_from_pdf(pdf_path)

    return chunk_text(
        text=text,
        document_id=document_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )