"""Tests for embedding generation utilities."""

from ml_knowledge_hub.embeddings.embedder import Embedder


def test_embedder_returns_embeddings():
    embedder = Embedder()

    texts = [
        "ResNet is a convolutional neural network.",
        "BERT is a transformer model.",
    ]

    embeddings = embedder.encode(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])