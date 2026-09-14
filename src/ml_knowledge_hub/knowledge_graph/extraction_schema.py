"""Schemas for extracting structured knowledge graph data."""

from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PROJECT = "project"
    MODEL = "model"
    DATASET = "dataset"
    EXPERIMENT = "experiment"
    METRIC = "metric"
    REPOSITORY = "repository"
    TASK = "task"
    DEPLOYMENT = "deployment"


class RelationType(str, Enum):
    USES_MODEL = "USES_MODEL"
    TRAINED_ON = "TRAINED_ON"
    EVALUATED_ON = "EVALUATED_ON"
    SOLVES = "SOLVES"
    TESTS = "TESTS"
    REPORTS = "REPORTS"
    STORED_IN = "STORED_IN"
    DEPLOYED_AS = "DEPLOYED_AS"
    USES_DATASET = "USES_DATASET"
    RELATED_TO = "RELATED_TO"


class KGEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: EntityType
    description: str | None = None
    properties: dict = Field(default_factory=dict)


class KGRelation(BaseModel):
    source_id: str
    relation_type: RelationType
    target_id: str
    evidence: str | None = None


class KGExtractionResult(BaseModel):
    entities: list[KGEntity] = Field(default_factory=list)
    relations: list[KGRelation] = Field(default_factory=list)