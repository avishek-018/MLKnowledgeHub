"""Run ingestion against sample data."""

from pathlib import Path

from ml_knowledge_hub.ingestion.pipeline import ingest_pdf


pdf_path = Path("data/raw/3731443.3771369 (2).pdf")

chunks = ingest_pdf(
    pdf_path=pdf_path,
    document_id="paper_001",
)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print("\n" + "=" * 80)
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Characters: {chunk.start_char} - {chunk.end_char}")
    print()
    print(chunk.text)