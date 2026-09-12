from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.assistant.service import KnowledgeAssistant
from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.corpus_loader import load_corpus
from ml_knowledge_hub.metadata.service import MetadataService
from ml_knowledge_hub.rag.generator import RAGGenerator
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


def main():
    load_dotenv()

    manifest_path = Path("data/raw/manifest.json")
    corpus_root = Path("data/raw/corpus")

    # 1. Load documents only for structured metadata queries
    documents = load_corpus(
        manifest_path=manifest_path,
        corpus_root=corpus_root,
    )

    print(f"Loaded documents: {len(documents)}")

    # 2. Structured metadata service
    metadata_service = MetadataService(documents)

    # 3. Query embedder
    embedder = Embedder()

    # 4. Open the existing persistent Qdrant index
    store = QdrantStore()

    indexed_points = store.count_points()

    print(f"Indexed points: {indexed_points}")

    if indexed_points == 0:
        raise RuntimeError(
            "Qdrant index is empty. Run "
            "`python scripts/index_corpus.py` first."
        )

    # 5. Grounded answer generator
    rag_generator = RAGGenerator()

    # 6. Unified assistant
    assistant = KnowledgeAssistant(
        metadata_service=metadata_service,
        vector_store=store,
        embedder=embedder,
        rag_generator=rag_generator,
    )

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