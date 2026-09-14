"""Tests for the knowledge graph processing pipeline."""

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
    KGEntity,
    KGExtractionResult,
    KGRelation,
    RelationType,
)
from ml_knowledge_hub.knowledge_graph.processing_pipeline import (
    KGProcessingPipeline,
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
    Predictable resolver for unit testing.

    This lets us test the pipeline without making an LLM API call.
    """

    def resolve(
        self,
        candidate,
        existing_entities,
    ):
        # Resolve SD 1.5 to an existing Stable Diffusion node.
        if candidate.name == "SD 1.5":
            return ResolutionDecision(
                action=ResolutionAction.MERGE,
                canonical_entity_id=(
                    "stable_diffusion_v1_5"
                ),
                confidence=0.99,
                reason="Known abbreviation.",
                matched_name=(
                    "Stable Diffusion v1.5"
                ),
            )

        # Treat the task as a new canonical entity.
        return ResolutionDecision(
            action=ResolutionAction.NEW,
            canonical_entity_id=None,
            confidence=0.99,
            reason="New entity.",
        )


def test_relation_ids_are_rewritten(tmp_path):

    # --------------------------------------------------
    # Create canonical registry
    # --------------------------------------------------
    registry = EntityRegistry(
        tmp_path / "entities.json"
    )

    # Stable Diffusion already exists in our canonical registry.
    registry.add_entity(
        EntityCandidate(
            entity_id="stable_diffusion_v1_5",
            name="Stable Diffusion v1.5",
            entity_type=EntityType.MODEL,
        )
    )

    # --------------------------------------------------
    # Build resolution pipeline
    # --------------------------------------------------
    resolution_pipeline = EntityResolutionPipeline(
        registry=registry,
        resolver=FakeResolver(),
    )

    processing_pipeline = KGProcessingPipeline(
        resolution_pipeline=resolution_pipeline,
        registry=registry,
    )

    # --------------------------------------------------
    # Simulate raw LLM KG extraction
    # --------------------------------------------------
    extraction = KGExtractionResult(
        entities=[
            KGEntity(
                entity_id="sd_1_5",
                name="SD 1.5",
                entity_type=EntityType.MODEL,
            ),
            KGEntity(
                entity_id="image_generation",
                name="Image Generation",
                entity_type=EntityType.TASK,
            ),
        ],
        relations=[
            KGRelation(
                source_id="sd_1_5",
                relation_type=RelationType.SOLVES,
                target_id="image_generation",
                evidence=(
                    "SD 1.5 performs image generation."
                ),
            )
        ],
    )

    result = processing_pipeline.process(
        extraction
    )

    # --------------------------------------------------
    # Verify identity resolution
    # --------------------------------------------------
    assert (
        result.entity_id_map["sd_1_5"]
        == "stable_diffusion_v1_5"
    )

    assert (
        result.entity_id_map[
            "image_generation"
        ]
        == "image_generation"
    )

    # --------------------------------------------------
    # Verify relation rewriting
    # --------------------------------------------------
    assert len(result.relations) == 1

    relation = result.relations[0]

    assert (
        relation.source_id
        == "stable_diffusion_v1_5"
    )

    assert (
        relation.target_id
        == "image_generation"
    )

    assert (
        relation.relation_type
        == RelationType.SOLVES
    )