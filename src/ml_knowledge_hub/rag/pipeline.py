"""Pipeline orchestration for retrieval-augmented generation."""

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.rag.generator import RAGGenerator


class RAGPipeline:
    def __init__(
        self,
        vector_store,
        embedder: Embedder,
        generator: RAGGenerator,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        query_embedding = self.embedder.encode(
            [question]
        )[0]

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
        )

        answer = self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        sources = [
            {
                "chunk_id": result.payload["chunk_id"],
                "document_id": result.payload["document_id"],
                "score": result.score,
            }
            for result in retrieved_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        } 