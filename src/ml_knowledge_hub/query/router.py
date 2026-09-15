"""Route user queries to retrieval and generation workflows."""

from dataclasses import dataclass
from typing import Literal


# Query types supported by the assistant.
#
# graph:
#   answer directly from Neo4j relationships
#
# hybrid:
#   combine graph structure with vector-retrieved evidence
QueryType = Literal[
    "metadata",
    "filtered_semantic",
    "semantic",
    "graph",
    "hybrid",
]


@dataclass
class QueryPlan:
    query_type: QueryType
    query: str
    asset_types: list[str] | None = None
    operation: str | None = None


def route_query(query: str) -> QueryPlan:
    q = query.lower().strip()

    # Structured metadata queries
    if "how many projects" in q:
        return QueryPlan(
            query_type="metadata",
            query=query,
            operation="count_projects",
        )

    if "list all projects" in q or "list the projects" in q:
        return QueryPlan(
            query_type="metadata",
            query=query,
            operation="list_projects",
        )

    if "how many model cards" in q:
        return QueryPlan(
            query_type="metadata",
            query=query,
            operation="count_model_cards",
        )

    # More specific filtered semantic queries first
    if "deployment note" in q:
        return QueryPlan(
            query_type="filtered_semantic",
            query=query,
            asset_types=["deployment_notes"],
        )

    if "postmortem" in q:
        return QueryPlan(
            query_type="filtered_semantic",
            query=query,
            asset_types=["postmortem"],
        )

    if "model card" in q:
        return QueryPlan(
            query_type="filtered_semantic",
            query=query,
            asset_types=["model_card"],
        )

    # Broader deployment-related query
    if "deployment" in q:
        return QueryPlan(
            query_type="filtered_semantic",
            query=query,
            asset_types=[
                "deployment_notes",
                "postmortem"
            ],
        )

    if "evaluation" in q:
        return QueryPlan(
            query_type="filtered_semantic",
            query=query,
            asset_types=[
                "evaluation_report",
                "experiment_report",
            ],
        )

        # --------------------------------------------------
    # Graph-structured questions
    # --------------------------------------------------

    # Questions about what models belong to a project.
    if (
        "which models" in q
        and "project" in q
    ):
        return QueryPlan(
            query_type="graph",
            query=query,
            operation="project_models",
        )

    # Questions about datasets connected to a model.
    if (
        "which datasets" in q
        and "model" in q
    ):
        return QueryPlan(
            query_type="graph",
            query=query,
            operation="model_datasets",
        )

    # Questions about metrics reported by a model.
    if (
        "which metrics" in q
        and "model" in q
    ):
        return QueryPlan(
            query_type="graph",
            query=query,
            operation="model_metrics",
        )

    # Questions asking for broader evidence around a model
    # benefit from both graph structure and document evidence.
    if (
        "tell me about" in q
        and "model" in q
    ):
        return QueryPlan(
            query_type="hybrid",
            query=query,
            operation="model_context",
        )
        
    # Default
    return QueryPlan(
        query_type="semantic",
        query=query,
    )