"""Smoke-test Neo4j entity operations."""


"""Test inserting one canonical entity into Neo4j."""

from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
    KGEntity,
)
from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)


def main():
    # --------------------------------------------------
    # Load Neo4j credentials from .env
    # --------------------------------------------------
    load_dotenv()

    store = Neo4jStore()

    try:
        # --------------------------------------------------
        # Create one simple canonical entity
        # --------------------------------------------------
        entity = KGEntity(
            entity_id="test_model_resnet50",
            name="ResNet50",
            entity_type=EntityType.MODEL,
            description="Test model node.",
            properties={
                "source": "manual_test"
            },
        )

        # --------------------------------------------------
        # Write it to Neo4j
        # --------------------------------------------------
        store.upsert_entity(
            entity
        )

        print(
            "Entity upsert successful."
        )

    finally:
        # Always close the driver cleanly.
        store.close()


if __name__ == "__main__":
    main()