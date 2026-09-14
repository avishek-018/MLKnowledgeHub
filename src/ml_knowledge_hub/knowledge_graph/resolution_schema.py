"""Schemas for resolving knowledge graph entities and relationships."""

"""Schemas for entity identity resolution."""

from enum import Enum

from pydantic import BaseModel, Field

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)


class ResolutionAction(str, Enum):
    MERGE = "merge"
    NEW = "new"
    REVIEW = "review"


class EntityCandidate(BaseModel):
    entity_id: str
    name: str
    entity_type: EntityType

    aliases: list[str] = Field(
        default_factory=list
    )

    description: str | None = None
    properties: dict = Field(
        default_factory=dict
    )


class ResolutionDecision(BaseModel):
    action: ResolutionAction

    canonical_entity_id: str | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    matched_name: str | None = None