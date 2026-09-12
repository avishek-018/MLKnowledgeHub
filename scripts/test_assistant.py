"""Smoke-test entry point for assistant workflows."""


from pathlib import Path

from ml_knowledge_hub.assistant.service import KnowledgeAssistant
from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.chunker import chunk_document
from ml_knowledge_hub.ingestion.corpus_loader import load_corpus
from ml_knowledge_hub.metadata.service import MetadataService
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore
from dotenv import load_dotenv

from ml_knowledge_hub.rag.generator import RAGGenerator

load_dotenv()
rag_generator = RAGGenerator()

def main():
    manifest_path = Path("data/raw/manifest.json")
    corpus_root = Path("data/raw/corpus")

    # 1. Load documents
    documents = load_corpus(
        manifest_path=manifest_path,
        corpus_root=corpus_root,
    )

    print(f"Loaded documents: {len(documents)}")

    # 2. Build metadata service
    metadata_service = MetadataService(documents)

    # 3. Chunk documents
    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

    print(f"Generated chunks: {len(all_chunks)}")

    # 4. Build embeddings
    embedder = Embedder()

    embeddings = embedder.encode(
        [chunk.text for chunk in all_chunks]
    )

    # 5. Build vector store
    store = QdrantStore()

    store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
    )

    # 6. Build assistant
    assistant = KnowledgeAssistant(
        metadata_service=metadata_service,
        vector_store=store,
        embedder=embedder,
        rag_generator=rag_generator,
    )

    # 7. Test questions
    questions = [
        "How many projects do we have?",
        "Please list all projects",
        "How many model cards do we have?",
        "Which model cards describe AI image detectors?",
        "Which projects use diffusion-based methods?",
        "What deployment issues were reported?",
    ]

    for question in questions:
        print("\n" + "=" * 100)
        print("QUESTION:")
        print(question)

        result = assistant.ask(question)

        print("\nRESULT TYPE:")
        print(result["type"])

        if result["type"] == "metadata":
            print("\nANSWER:")
            print(result["answer"])

        elif result["type"] == "semantic":
            print("\nFILTER:")
            print(result["filter"])

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")

            for source in result["sources"]:
                print(
                    f"{source['source_id']} | "
                    f"{source['title']} | "
                    f"{source['project_id']} | "
                    f"{source['asset_type']} | "
                    f"score={source['score']:.4f}"
                )


if __name__ == "__main__":
    main()
