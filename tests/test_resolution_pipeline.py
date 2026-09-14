"""Tests for the knowledge graph resolution pipeline."""

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.resolution_pipeline import (
    EntityResolutionPipeline,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
    ResolutionAction,
    ResolutionDecision,
)


class FakeResolver:
    """
    Small fake resolver used for unit testing.

    This avoids making real LLM/API calls during pytest.
    """

    def __init__(
        self,
        decision: ResolutionDecision,
    ):
        self.decision = decision

    def resolve(
        self,
        candidate,
        existing_entities,
    ):
        return self.decision


def test_new_entity_is_added(tmp_path):
    # Create an isolated temporary entity registry.
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    # Simulate the resolver deciding that this is a new entity.
    fake_resolver = FakeResolver(
        ResolutionDecision(
            action=ResolutionAction.NEW,
            canonical_entity_id=None,
            confidence=0.99,
            reason="Distinct model.",
        )
    )

    pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=fake_resolver,
    )

    candidate = EntityCandidate(
        entity_id="temporary_id",
        name="Stable Diffusion 3.5",
        entity_type=EntityType.MODEL,
    )

    pipeline.resolve_entity(
        candidate
    )

    # The candidate should now exist as a canonical entity
    # using a normalized stable entity ID.
    entity = registry.get_entity(
        "stable_diffusion_3_5"
    )

    assert entity is not None
    assert entity.name == "Stable Diffusion 3.5"


def test_merge_adds_alias(tmp_path):
    # Create an isolated registry with one known canonical entity.
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    canonical = EntityCandidate(
        entity_id="stable_diffusion_v1_5",
        name="Stable Diffusion v1.5",
        entity_type=EntityType.MODEL,
    )

    registry.add_entity(
        canonical
    )

    # Simulate a confident LLM merge decision.
    fake_resolver = FakeResolver(
        ResolutionDecision(
            action=ResolutionAction.MERGE,
            canonical_entity_id="stable_diffusion_v1_5",
            confidence=0.99,
            reason="Same model and version.",
            matched_name="Stable Diffusion v1.5",
        )
    )

    pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=fake_resolver,
    )

    candidate = EntityCandidate(
        entity_id="sd_1_5",
        name="SD 1.5",
        entity_type=EntityType.MODEL,
    )

    pipeline.resolve_entity(
        candidate
    )

    updated = registry.get_entity(
        "stable_diffusion_v1_5"
    )

    # The new name should now be stored as an alias.
    assert "SD 1.5" in updated.aliases


def test_review_does_not_create_entity(tmp_path):
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    # Simulate an ambiguous LLM decision.
    fake_resolver = FakeResolver(
        ResolutionDecision(
            action=ResolutionAction.REVIEW,
            canonical_entity_id=None,
            confidence=0.97,
            reason="Identity is ambiguous.",
        )
    )

    pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=fake_resolver,
    )

    candidate = EntityCandidate(
        entity_id="sd_new",
        name="SD New",
        entity_type=EntityType.MODEL,
    )

    pipeline.resolve_entity(
        candidate
    )

    # REVIEW candidates should not become canonical entities.
    assert registry.get_entity(
        "sd_new"
    ) is None


def test_new_project_preserves_authoritative_project_id(
    tmp_path,
):
    """
    Project identifiers come from the asset registry and
    must not be regenerated from display names.
    """

    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    # Simulate a resolver deciding that this project
    # has not been seen before.
    fake_resolver = FakeResolver(
        ResolutionDecision(
            action=ResolutionAction.NEW,
            canonical_entity_id=None,
            confidence=0.99,
            reason="New project.",
        )
    )

    pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=fake_resolver,
    )

    candidate = EntityCandidate(
        # This is the authoritative project ID.
        entity_id="nyuad_ai_image_detector",

        # The human-readable name is intentionally different.
        name="NYUAD AI-generated Images Detector",

        entity_type=EntityType.PROJECT,
    )

    pipeline.resolve_entity(
        candidate
    )

    # The registry must preserve the authoritative project ID.
    project = registry.get_entity(
        "nyuad_ai_image_detector"
    )

    assert project is not None

    assert (
        project.name
        == "NYUAD AI-generated Images Detector"
    )

    # A name-derived project ID must NOT have been created.
    assert registry.get_entity(
        "nyuad_ai_generated_images_detector"
    ) is None


