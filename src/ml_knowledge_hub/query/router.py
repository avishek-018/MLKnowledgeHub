"""Route user queries to retrieval and generation workflows."""

from dataclasses import dataclass
from typing import Literal


QueryType = Literal[
    "metadata",
    "filtered_semantic",
    "semantic",
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

    # Default
    return QueryPlan(
        query_type="semantic",
        query=query,
    )