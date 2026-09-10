from pathlib import Path

from ml_knowledge_hub.ingestion.pdf_parser import extract_text_from_pdf


pdf_path = Path("data/raw/3731443.3771369 (2).pdf")

text = extract_text_from_pdf(pdf_path)

print(f"Extracted characters: {len(text)}")
print()
print(text[:3000])