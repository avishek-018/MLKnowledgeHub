"""Tests for entity resolution schemas."""

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
    ResolutionAction,
    ResolutionDecision,
    
)
from ml_knowledge_hub.knowledge_graph.entity_resolver import EntityResolver

def test_entity_candidate():
    entity = EntityCandidate(
        entity_id="stable_diffusion_v1_5",
        name="Stable Diffusion v1.5",
        entity_type=EntityType.MODEL,
        aliases=[
            "Stable Diffusion 1.5",
            "SD 1.5",
        ],
    )

    assert entity.entity_type == EntityType.MODEL
    assert "SD 1.5" in entity.aliases


def test_resolution_decision():
    decision = ResolutionDecision(
        action=ResolutionAction.MERGE,
        canonical_entity_id="stable_diffusion_v1_5",
        confidence=0.98,
        reason=(
            "The candidate refers to the same "
            "model and version."
        ),
    )

    assert decision.action == ResolutionAction.MERGE
    assert decision.confidence == 0.98

from ml_knowledge_hub.knowledge_graph.normalizer import (
    normalize_entity_id,
)


def test_version_normalization():
    assert (
        normalize_entity_id(
            "Stable Diffusion v1.5"
        )
        ==
        "stable_diffusion_1_5"
    )

    assert (
        normalize_entity_id(
            "Stable Diffusion 1.5"
        )
        ==
        "stable_diffusion_1_5"
    )


def test_different_entity_types_do_not_merge():
    """
    A project and model with similar names must remain
    separate canonical entities.
    """

    project = EntityCandidate(
        entity_id="nyuad_ai_image_detector",
        name="NYUAD AI Image Detector",
        entity_type=EntityType.PROJECT,
    )

    model = EntityCandidate(
        entity_id="nyuad_ai_generated_images_detector",
        name="NYUAD AI-generated Images Detector",
        entity_type=EntityType.MODEL,
    )

    resolver = EntityResolver()

    # Since the only existing entity is a PROJECT and
    # the new candidate is a MODEL, there are no
    # compatible entities to merge against.
    decision = resolver.resolve(
        candidate=model,
        existing_entities=[project],
    )

    assert decision.action == ResolutionAction.NEW
    assert decision.canonical_entity_id is None


def test_different_project_ids_do_not_merge():
    """
    Similar project names must not cause different
    authoritative project IDs to merge.
    """

    existing_project = EntityCandidate(
        entity_id="nyuad_ai_image_detector",
        name="NYUAD AI-generated Images Detector",
        entity_type=EntityType.PROJECT,
    )

    candidate = EntityCandidate(
        entity_id="cnn_detection",
        name="CNN Detection",
        entity_type=EntityType.PROJECT,
    )

    resolver = EntityResolver()

    decision = resolver.resolve(
        candidate=candidate,
        existing_entities=[existing_project],
    )

    assert decision.action == ResolutionAction.NEW
    assert decision.confidence == 1.0


def test_same_project_id_merges():
    """
    Repeated occurrences of the same project ID
    should resolve deterministically.
    """

    existing_project = EntityCandidate(
        entity_id="cnn_detection",
        name="CNN Detection",
        entity_type=EntityType.PROJECT,
    )

    candidate = EntityCandidate(
        entity_id="cnn_detection",
        name="CNNDetection",
        entity_type=EntityType.PROJECT,
    )

    resolver = EntityResolver()

    decision = resolver.resolve(
        candidate=candidate,
        existing_entities=[existing_project],
    )

    assert decision.action == ResolutionAction.MERGE
    assert (
        decision.canonical_entity_id
        == "cnn_detection"
    )