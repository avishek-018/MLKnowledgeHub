"""Tests for canonical knowledge graph identifiers."""

from ml_knowledge_hub.knowledge_graph.canonical_ids import (
    canonical_id_for_new_entity,
)
from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
)


def test_cross_type_id_collision(tmp_path):
    """
    A model and project may have the same normalized name,
    but they must receive different canonical IDs.
    """

    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    # Existing project already owns this ID.
    project = EntityCandidate(
        entity_id="diffusion_detector",
        name="Diffusion Detector",
        entity_type=EntityType.PROJECT,
    )

    registry.add_entity(project)

    # A distinct MODEL happens to have the same name.
    model = EntityCandidate(
        entity_id="raw_model_id",
        name="Diffusion Detector",
        entity_type=EntityType.MODEL,
    )

    canonical_id = canonical_id_for_new_entity(
        candidate=model,
        registry=registry,
    )

    # It must not steal the project's identifier.
    assert canonical_id == (
        "model__diffusion_detector"
    )


def test_project_preserves_authoritative_id(tmp_path):
    """
    Project IDs must continue to come from the asset
    registry rather than the display name.
    """

    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    project = EntityCandidate(
        entity_id="dm_image_detection",
        name="Detection of Diffusion Images",
        entity_type=EntityType.PROJECT,
    )

    canonical_id = canonical_id_for_new_entity(
        candidate=project,
        registry=registry,
    )

    assert canonical_id == "dm_image_detection"