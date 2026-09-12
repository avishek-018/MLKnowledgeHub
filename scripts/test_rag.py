"""Smoke-test entry point for RAG workflows."""


from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.pipeline import ingest_pdf
from ml_knowledge_hub.rag.generator import RAGGenerator
from ml_knowledge_hub.rag.pipeline import RAGPipeline
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


load_dotenv()


pdf_path = Path("data/raw/3731443.3771369 (2).pdf")

# Ingest
chunks = ingest_pdf(
    pdf_path=pdf_path,
    document_id="paper_001",
)

# Embed
embedder = Embedder()

embeddings = embedder.encode(
    [chunk.text for chunk in chunks]
)

# Store
store = QdrantStore()

store.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
)

# RAG
generator = RAGGenerator()

rag = RAGPipeline(
    vector_store=store,
    embedder=embedder,
    generator=generator,
)

question = "What is the accuracy?"

result = rag.ask(
    question=question,
    top_k=5,
)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nRETRIEVED SOURCES:")

for source in result["sources"]:
    print(
        f"{source['chunk_id']} "
        f"(score={source['score']:.4f})"
    )