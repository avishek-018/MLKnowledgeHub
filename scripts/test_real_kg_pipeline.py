"""Run the real KG extraction + entity resolution pipeline."""

from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.ingestion.parser import extract_text
from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.entity_resolver import (
    EntityResolver,
)
from ml_knowledge_hub.knowledge_graph.extractor import (
    KGExtractor,
)
from ml_knowledge_hub.knowledge_graph.processing_pipeline import (
    KGProcessingPipeline,
)
from ml_knowledge_hub.knowledge_graph.resolution_pipeline import (
    EntityResolutionPipeline,
)

from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)


def main():
    # --------------------------------------------------
    # Load environment variables
    # --------------------------------------------------
    load_dotenv()

    # --------------------------------------------------
    # 1. Read one real artifact
    # --------------------------------------------------
    file_path = Path(
        "data/raw/corpus/model_cards/"
        "nyuad_ai_generated_images_detector__MODEL_CARD.md"
    )

    text = extract_text(file_path)

    # --------------------------------------------------
    # 2. Extract raw KG entities + relations using LLM
    # --------------------------------------------------
    extractor = KGExtractor()

    raw_result = extractor.extract(
        text=text,
        project_id="nyuad_ai_image_detector",
        asset_type="model_card",
        title=(
            "NYUAD AI-generated Images Detector "
            "Model Card"
        ),
    )

    print("\nRAW KG")
    print("======")

    print("\nEntities:")
    for entity in raw_result.entities:
        print(
            f"- {entity.entity_id} | "
            f"{entity.entity_type.value} | "
            f"{entity.name}"
        )

    print("\nRelations:")
    for relation in raw_result.relations:
        print(
            f"- {relation.source_id} "
            f"--{relation.relation_type.value}--> "
            f"{relation.target_id}"
        )

    # --------------------------------------------------
    # 3. Create persistent canonical entity registry
    # --------------------------------------------------
    registry = EntityRegistry(
        "data/knowledge_graph/entities.json"
    )

    # --------------------------------------------------
    # 4. Create entity resolver
    # --------------------------------------------------
    resolver = EntityResolver()

    # --------------------------------------------------
    # 5. Create entity resolution pipeline
    # --------------------------------------------------
    resolution_pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=resolver,
    )

    # --------------------------------------------------
    # 6. Create KG processing pipeline
    # --------------------------------------------------
    processing_pipeline = KGProcessingPipeline(
        resolution_pipeline=resolution_pipeline,
        registry=registry,
    )

    # --------------------------------------------------
    # 7. Resolve raw entities and rewrite relations
    # --------------------------------------------------
    resolved_result = processing_pipeline.process(
        raw_result
    )

    # --------------------------------------------------
    # 8. Write resolved KG into Neo4j
    # --------------------------------------------------
    neo4j_store = Neo4jStore()

    try:
        neo4j_store.write_result(
            resolved_result
        )

        print(
            "\nResolved KG written to Neo4j."
        )

    finally:
        neo4j_store.close()

    print("\n\nRESOLVED KG")
    print("===========")

    print("\nEntity ID mapping:")

    for raw_id, canonical_id in (
        resolved_result.entity_id_map.items()
    ):
        print(
            f"- {raw_id} -> {canonical_id}"
        )

    print("\nCanonical entities:")

    for entity in resolved_result.entities:
        print(
            f"- {entity.entity_id} | "
            f"{entity.entity_type.value} | "
            f"{entity.name}"
        )

    print("\nCanonical relations:")

    for relation in resolved_result.relations:
        print(
            f"- {relation.source_id} "
            f"--{relation.relation_type.value}--> "
            f"{relation.target_id}"
        )

        if relation.evidence:
            print(
                f"  Evidence: {relation.evidence}"
            )

    # --------------------------------------------------
    # 9. Show unresolved entities
    # --------------------------------------------------
    if resolved_result.unresolved_entity_ids:
        print("\nUnresolved entities:")

        for entity_id in (
            resolved_result.unresolved_entity_ids
        ):
            print(
                f"- {entity_id}"
            )

    else:
        print("\nNo unresolved entities.")


if __name__ == "__main__":
    main()