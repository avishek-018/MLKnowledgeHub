"""Tests for the Qdrant vector store integration."""

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.chunker import chunk_text
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


def test_qdrant_store_returns_relevant_chunks():
    text = """
    SynthBuster detects synthetic images using frequency-domain features.
    It extracts FFT-based information from images.

    ResNet is a convolutional neural network architecture used for image classification.

    The weather is sunny and warm today.
    """

    chunks = chunk_text(
        text=text,
        document_id="test_doc",
        chunk_size=100,
        overlap=20,
    )

    embedder = Embedder()

    texts = [chunk.text for chunk in chunks]

    embeddings = embedder.encode(texts)

    store = QdrantStore()

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = "frequency features for detecting synthetic images"

    query_embedding = embedder.encode([query])[0]

    results = store.search(
        query_embedding=query_embedding,
        limit=3,
    )

    assert len(results) > 0
    assert "frequency" in results[0].payload["text"].lower()