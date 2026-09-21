"""Run an interactive demonstration of the knowledge assistant."""

from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.assistant.service import KnowledgeAssistant
from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.corpus_loader import load_corpus
from ml_knowledge_hub.metadata.service import MetadataService
from ml_knowledge_hub.rag.generator import RAGGenerator
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore
from ml_knowledge_hub.registry.service import AssetRegistry
from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)
from ml_knowledge_hub.knowledge_graph.query_service import (
    GraphQueryService,
)

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)

from pprint import pprint

def main():
    load_dotenv()

    # --------------------------------------------------
    # Neo4j graph query layer
    # --------------------------------------------------
    neo4j_store = Neo4jStore()

    graph_service = GraphQueryService(
        store=neo4j_store
    )

    entity_registry = EntityRegistry(
        "data/knowledge_graph/entities.json"
    )
    

    manifest_path = Path("data/raw/manifest.json")
    corpus_root = Path("data/raw/corpus")

    # # 1. Load documents only for structured metadata queries
    # documents = load_corpus(
    #     manifest_path=manifest_path,
    #     corpus_root=corpus_root,
    # )

    # print(f"Loaded documents: {len(documents)}")

    # 2. Structured metadata service
    # metadata_service = MetadataService(documents)
    registry = AssetRegistry(
        manifest_path
    )

    metadata_service = MetadataService(
        registry
    )


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
        graph_service=graph_service,
        entity_registry=entity_registry,
        )

    questions = [
        "How many projects do we have?",
        "Please list all projects",
        "How many model cards do we have?",
        "Which model cards describe AI image detectors?",
        "Which projects use diffusion-based methods?",
        "What deployment issues were reported?",
        "Which model cards discuss fraud detection?",
        "Which models are used by the NYUAD project?",
        "Which datasets are used by the AI-generated images detector model?",
        "Which metrics are reported by the AI-generated images detector model?",
        "Tell me about the AI-generated images detector model.",
        "What do we know about GenImage as a project?",
    ]

    for question in questions:
        print("\n" + "=" * 100)
        print("QUESTION:")
        print(question)

        result = assistant.ask(question)

        print("\nRESULT TYPE:")
        print(result["type"])

        # --------------------------------------------------
        # Print the answer for every result type.
        # --------------------------------------------------
        print("\nANSWER:")
        answer = result.get("answer")

        if isinstance(answer, (list, dict)):
            pprint(
                answer,
                sort_dicts=False,
            )
        else:
            print(answer)

        # --------------------------------------------------
        # Semantic / filtered semantic queries may include
        # vector-search filters and document sources.
        # --------------------------------------------------
        if result["type"] == "semantic":

            if "filter" in result:
                print("\nFILTER:")
                print(result["filter"])

            print("\nSOURCES:")

            for source in result.get("sources", []):
                print(
                    f"{source['source_id']} | "
                    f"{source['title']} | "
                    f"{source['project_id']} | "
                    f"{source['asset_type']} | "
                    f"score={source['score']:.4f}"
                )

        # --------------------------------------------------
        # Graph queries return structured Neo4j results.
        # Show the raw graph result too for debugging.
        # --------------------------------------------------
        elif result["type"] == "graph":
            print("\nGRAPH RESULTS:")

            for item in result.get("raw", []):
                print(item)

        elif result["type"] == "hybrid":
            print("\nGRAPH CONTEXT:")
            print(result.get("graph_context"))

            print("\nSOURCES:")

            for source in result.get(
                "sources",
                [],
            ):
                print(
                    f"{source['source_id']} | "
                    f"{source['title']} | "
                    f"{source['project_id']} | "
                    f"{source['asset_type']} | "
                    f"score={source['score']:.4f}"
                )
    # --------------------------------------------------
    # Close Neo4j after all questions are processed.
    # --------------------------------------------------
    neo4j_store.close()


if __name__ == "__main__":
    main()
