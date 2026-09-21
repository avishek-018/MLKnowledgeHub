"""Tests for knowledge graph schemas."""

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
    KGEntity,
    KGExtractionResult,
    KGRelation,
    RelationType,
)


def test_kg_extraction_schema():
    model = KGEntity(
        entity_id="model_resnet50",
        name="ResNet50",
        entity_type=EntityType.MODEL,
    )

    dataset = KGEntity(
        entity_id="dataset_imagenet",
        name="ImageNet",
        entity_type=EntityType.DATASET,
    )

    relation = KGRelation(
        source_id="model_resnet50",
        relation_type=RelationType.TRAINED_ON,
        target_id="dataset_imagenet",
        evidence="ResNet50 was trained on ImageNet.",
    )

    result = KGExtractionResult(
        entities=[model, dataset],
        relations=[relation],
    )

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == RelationType.TRAINED_ON