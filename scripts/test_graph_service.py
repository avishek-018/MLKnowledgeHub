"""Smoke-test knowledge graph service workflows."""


"""Test the high-level graph query service."""

from pprint import pprint

from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)
from ml_knowledge_hub.knowledge_graph.query_service import (
    GraphQueryService,
)


def main():
    # --------------------------------------------------
    # Load Neo4j credentials.
    # --------------------------------------------------
    load_dotenv()

    store = Neo4jStore()

    graph_service = GraphQueryService(
        store=store
    )

    try:
        # --------------------------------------------------
        # Retrieve structured context for one real model.
        # --------------------------------------------------
        context = graph_service.get_model_context(
            "ai_generated_images_detector"
        )

        print()
        print("=" * 70)
        print("GRAPH MODEL CONTEXT")
        print("=" * 70)

        pprint(
            context,
            sort_dicts=False,
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()