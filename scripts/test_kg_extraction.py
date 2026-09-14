"""Smoke-test entry point for knowledge graph extraction."""


from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.ingestion.parser import extract_text
from ml_knowledge_hub.knowledge_graph.extractor import KGExtractor


def main():
    load_dotenv()

    file_path = Path(
        "data/raw/corpus/model_cards/"
        "nyuad_ai_generated_images_detector__MODEL_CARD.md"
    )

    text = extract_text(file_path)

    extractor = KGExtractor()

    result = extractor.extract(
        text=text,
        project_id="nyuad_ai_image_detector",
        asset_type="model_card",
        title="NYUAD AI-generated Images Detector Model Card",
    )

    print("\nENTITIES:")

    for entity in result.entities:
        print(
            f"- {entity.entity_id} | "
            f"{entity.entity_type.value} | "
            f"{entity.name}"
        )

    print("\nRELATIONS:")

    for relation in result.relations:
        print(
            f"- {relation.source_id} "
            f"--{relation.relation_type.value}--> "
            f"{relation.target_id}"
        )

        if relation.evidence:
            print(
                f"  Evidence: {relation.evidence}"
            )


if __name__ == "__main__":
    main()
