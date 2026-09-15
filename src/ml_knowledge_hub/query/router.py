"""Query routing adapter for the ML Knowledge Hub."""

from dataclasses import dataclass
from typing import Literal

from ml_knowledge_hub.agents.orchestrator import (
    AgentOrchestrator,
)


QueryType = Literal[
    "metadata",
    "filtered_semantic",
    "semantic",
    "graph",
    "hybrid",
]


@dataclass
class QueryPlan:
    """
    Execution plan used by the existing KnowledgeAssistant.

    This is intentionally kept separate from QueryPlanLLM.

    QueryPlanLLM:
        used by the agentic reasoning layer.

    QueryPlan:
        used by the deterministic execution layer.
    """

    query_type: QueryType
    query: str

    asset_types: list[str] | None = None
    operation: str | None = None

    entity_mention: str | None = None
    entity_type: str | None = None

    target_entity_type: str | None = None


class QueryRouter:
    """
    Convert a natural-language query into an executable QueryPlan.

    Planning is delegated to the agentic orchestration layer:

        Planner
            ↓
        Evaluator
            ↓
        Revision if needed
            ↓
        final plan

    The router then converts that typed LLM plan into the format
    expected by the existing KnowledgeAssistant.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator | None = None,
    ):
        # --------------------------------------------------
        # Dependency injection allows us to substitute a fake
        # orchestrator during unit tests.
        # --------------------------------------------------
        self.orchestrator = (
            orchestrator
            if orchestrator is not None
            else AgentOrchestrator()
        )

    def route(
        self,
        query: str,
    ) -> QueryPlan:
        """
        Produce the final executable query plan.
        """

        # --------------------------------------------------
        # Run the complete agentic planning workflow.
        # --------------------------------------------------
        orchestration_result = self.orchestrator.create_plan(
            query=query
        )

        final_plan = orchestration_result[
            "final_plan"
        ]

        # --------------------------------------------------
        # Convert typed asset enums into the string values
        # expected by Qdrant filtering.
        # --------------------------------------------------
        asset_types = (
            [
                asset_type.value
                for asset_type in final_plan.asset_types
            ]
            if final_plan.asset_types
            else None
        )

        # --------------------------------------------------
        # Convert entity enum values back into strings used
        # by the current execution layer.
        # --------------------------------------------------
        entity_type = (
            final_plan.entity_type.value
            if final_plan.entity_type
            else None
        )

        target_entity_type = (
            final_plan.target_entity_type.value
            if final_plan.target_entity_type
            else None
        )

        # --------------------------------------------------
        # Convert the agent-generated plan into the existing
        # execution-plan representation.
        # --------------------------------------------------
        return QueryPlan(
            query_type=final_plan.query_type.value,
            query=query,
            operation=final_plan.operation.value,
            asset_types=asset_types,
            entity_mention=final_plan.entity_mention,
            entity_type=entity_type,
            target_entity_type=target_entity_type,
        )