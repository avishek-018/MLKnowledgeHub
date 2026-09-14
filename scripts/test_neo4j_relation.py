"""Smoke-test Neo4j relationship operations."""


"""Test inserting two entities and one relation into Neo4j."""

from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
    KGEntity,
    KGRelation,
    RelationType,
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
        # Create a project node
        # --------------------------------------------------
        project = KGEntity(
            entity_id="test_project",
            name="Test ML Project",
            entity_type=EntityType.PROJECT,
            description="Temporary Neo4j relation test.",
        )

        # --------------------------------------------------
        # Create a model node
        # --------------------------------------------------
        model = KGEntity(
            entity_id="test_model_resnet50",
            name="ResNet50",
            entity_type=EntityType.MODEL,
            description="Test model node.",
        )

        # --------------------------------------------------
        # Upsert both entities first
        # --------------------------------------------------
        store.upsert_entity(project)
        store.upsert_entity(model)

        # --------------------------------------------------
        # Create the relationship
        #
        # Test ML Project --USES_MODEL--> ResNet50
        # --------------------------------------------------
        relation = KGRelation(
            source_id="test_project",
            relation_type=RelationType.USES_MODEL,
            target_id="test_model_resnet50",
            evidence=(
                "Temporary test showing that the project "
                "uses the ResNet50 model."
            ),
        )

        # --------------------------------------------------
        # Write the relationship
        # --------------------------------------------------
        store.upsert_relation(relation)

        print(
            "Relation upsert successful."
        )

    finally:
        # Always close the Neo4j driver.
        store.close()


if __name__ == "__main__":
    main()

