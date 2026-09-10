"""PDF parsing utilities."""

from pathlib import Path

import fitz


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text from all pages.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    text_parts = []

    with fitz.open(pdf_path) as document:
        for page in document:
            text = page.get_text("text")
            if text:
                text_parts.append(text)

    return "\n".join(text_parts).strip()