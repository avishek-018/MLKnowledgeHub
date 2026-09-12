"""Index a document corpus into the configured vector store."""

from pathlib import Path

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.chunker import chunk_document
from ml_knowledge_hub.ingestion.corpus_loader import load_corpus
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


def main():
    manifest_path = Path("data/raw/manifest.json")
    corpus_root = Path("data/raw/corpus")

    # 1. Load corpus
    documents = load_corpus(
        manifest_path=manifest_path,
        corpus_root=corpus_root,
    )

    print(f"Loaded documents: {len(documents)}")

    # 2. Chunk documents
    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

    print(f"Generated chunks: {len(all_chunks)}")

    # 3. Create embeddings
    embedder = Embedder()

    embeddings = embedder.encode(
        [chunk.text for chunk in all_chunks]
    )

    # 4. Index in Qdrant
    store = QdrantStore()

    store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
    )

    print(f"Indexed chunks: {len(all_chunks)}")
    print("Indexing complete.")


if __name__ == "__main__":
    main()