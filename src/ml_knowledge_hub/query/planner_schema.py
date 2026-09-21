"""Schemas for LLM-based query planning."""

from enum import Enum

from pydantic import BaseModel


class PlannedQueryType(str, Enum):
    CONVERSATION = "conversation"
    METADATA = "metadata"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    HYBRID = "hybrid"


class PlannedOperation(str, Enum):
    # Conversation / scope handling
    GREETING = "greeting"
    HELP = "help"
    ACKNOWLEDGEMENT = "acknowledgement"
    GOODBYE = "goodbye"
    OUT_OF_SCOPE = "out_of_scope"

    # Metadata
    COUNT_PROJECTS = "count_projects"
    LIST_PROJECTS = "list_projects"
    COUNT_MODEL_CARDS = "count_model_cards"

    # Graph
    PROJECT_MODELS = "project_models"
    MODEL_DATASETS = "model_datasets"
    MODEL_METRICS = "model_metrics"

    # Hybrid
    MODEL_CONTEXT = "model_context"
    PROJECT_CONTEXT = "project_context"

    # Semantic fallback
    SEMANTIC_SEARCH = "semantic_search"

class PlannedAssetType(str, Enum):
    """
    Artifact types that can be used to constrain
    semantic retrieval.
    """

    PAPER = "paper"
    REPOSITORY_README = "repository_readme"
    MODEL_CARD = "model_card"
    DATASET_CARD = "dataset_card"
    DATASET_METADATA = "dataset_metadata"
    REPRODUCIBILITY_REPORT = "reproducibility_report"
    DEPLOYMENT_NOTES = "deployment_notes"
    POSTMORTEM = "postmortem"
    EVALUATION_REPORT = "evaluation_report"
    EXPERIMENT_REPORT = "experiment_report"
    PROJECT_BRIEF = "project_brief"

class PlannedEntityType(str, Enum):
    PROJECT = "project"
    MODEL = "model"
    DATASET = "dataset"
    METRIC = "metric"
    REPOSITORY = "repository"
    TASK = "task"
    EXPERIMENT = "experiment"
    DEPLOYMENT = "deployment"


class QueryPlanLLM(BaseModel):
    query_type: PlannedQueryType
    operation: PlannedOperation

    # Entity explicitly mentioned in the query.
    entity_mention: str | None = None

    # Type of the entity the query starts from.
    entity_type: PlannedEntityType | None = None

    # Type of entity the user wants returned.
    target_entity_type: PlannedEntityType | None = None

    # Optional document-type restriction.
    asset_types: list[PlannedAssetType] | None = None
