"""Tests for the knowledge graph entity registry."""

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
)


def test_add_and_get_entity(tmp_path):
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    entity = EntityCandidate(
        entity_id="stable_diffusion_v1_5",
        name="Stable Diffusion v1.5",
        entity_type=EntityType.MODEL,
        aliases=[
            "Stable Diffusion 1.5",
        ],
    )

    registry.add_entity(entity)

    loaded = registry.get_entity(
        "stable_diffusion_v1_5"
    )

    assert loaded is not None
    assert loaded.name == "Stable Diffusion v1.5"


def test_add_alias(tmp_path):
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    entity = EntityCandidate(
        entity_id="stable_diffusion_xl",
        name="Stable Diffusion XL",
        entity_type=EntityType.MODEL,
    )

    registry.add_entity(entity)

    registry.add_alias(
        "stable_diffusion_xl",
        "SDXL",
    )

    loaded = registry.get_entity(
        "stable_diffusion_xl"
    )

    assert "SDXL" in loaded.aliases


def test_duplicate_entity_rejected(tmp_path):
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    entity = EntityCandidate(
        entity_id="resnet50",
        name="ResNet50",
        entity_type=EntityType.MODEL,
    )

    registry.add_entity(entity)

    try:
        registry.add_entity(entity)
        assert False
    except ValueError:
        assert True