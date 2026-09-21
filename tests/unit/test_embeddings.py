"""Smoke-test entry point for embedding workflows."""


import numpy as np

from ml_knowledge_hub.embeddings.embedder import Embedder


embedder = Embedder()

texts = [
    "The model uses Fourier frequency features for fake image detection.",
    "FFT-based features are extracted from synthetic images.",
    "The weather is sunny today.",
]

embeddings = embedder.encode(texts)

query = "frequency features for detecting generated images"
query_embedding = embedder.encode([query])[0]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


for text, embedding in zip(texts, embeddings):
    similarity = cosine_similarity(
        query_embedding,
        embedding,
    )

    print(f"{similarity:.4f} | {text}")