"""Search the vector store using a sample PDF query."""


from pathlib import Path

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.pipeline import ingest_pdf
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


pdf_path = Path("data/raw/3731443.3771369 (2).pdf")

document_id = "paper_001"

# 1. Ingest PDF
chunks = ingest_pdf(
    pdf_path=pdf_path,
    document_id=document_id,
)

print(f"Total chunks: {len(chunks)}")

# 2. Embed chunks
embedder = Embedder()

chunk_texts = [chunk.text for chunk in chunks]
embeddings = embedder.encode(chunk_texts)

# 3. Store in Qdrant
store = QdrantStore()

store.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
)

# 4. Query
query = "What datasets are used in this paper?"

query_embedding = embedder.encode([query])[0]

results = store.search(
    query_embedding=query_embedding,
    limit=5,
)

# 5. Print results
print("\nQUERY:")
print(query)

print("\nTOP RESULTS:")

for rank, result in enumerate(results, start=1):
    print("\n" + "=" * 80)
    print(f"Rank: {rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Chunk ID: {result.payload['chunk_id']}")
    print()
    print(result.payload["text"][:1200])