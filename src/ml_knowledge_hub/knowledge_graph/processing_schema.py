"""Schemas for knowledge graph processing stages."""

"""Schemas for resolved knowledge-graph data."""

from pydantic import BaseModel, Field

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    KGEntity,
    KGRelation,
)


class ResolvedKGResult(BaseModel):
    """
    Final KG data after entity identity resolution.

    entity_id_map maps raw extractor IDs to canonical IDs.

    Example:
        "sd_1_5" -> "stable_diffusion_v1_5"
    """

    entities: list[KGEntity] = Field(
        default_factory=list
    )

    relations: list[KGRelation] = Field(
        default_factory=list
    )

    entity_id_map: dict[str, str] = Field(
        default_factory=dict
    )

    unresolved_entity_ids: list[str] = Field(
        default_factory=list
    )