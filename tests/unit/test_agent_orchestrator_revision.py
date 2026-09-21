"""Test that the orchestrator performs a revision when the initial plan is bad."""

from dotenv import load_dotenv

from ml_knowledge_hub.agents.orchestrator import AgentOrchestrator
from ml_knowledge_hub.agents.evaluator_agent import EvaluatorAgent
from ml_knowledge_hub.agents.revision_agent import RevisionAgent
from ml_knowledge_hub.query.planner_schema import (
    PlannedEntityType,
    PlannedOperation,
    PlannedQueryType,
    QueryPlanLLM,
)


class FakeBadPlanner:
    """
    Deliberately returns an incorrect plan.

    This lets us verify that the evaluator and revision
    agents can recover from a bad initial decision.
    """

    def plan(
        self,
        query: str,
    ) -> QueryPlanLLM:
        return QueryPlanLLM(
            query_type=PlannedQueryType.SEMANTIC,
            operation=PlannedOperation.SEMANTIC_SEARCH,
            entity_mention="NYUAD",
            entity_type=PlannedEntityType.PROJECT,
            target_entity_type=PlannedEntityType.MODEL,
            asset_types=None,
        )


def main():
    load_dotenv()

    orchestrator = AgentOrchestrator(
        planner=FakeBadPlanner(),
        evaluator=EvaluatorAgent(),
        revision_agent=RevisionAgent(),
    )

    question = "What models belong to NYUAD?"

    result = orchestrator.create_plan(
        query=question
    )

    print("=" * 100)
    print("QUESTION:")
    print(question)

    print("\nINITIAL PLAN:")
    print(
        result["initial_plan"].model_dump(
            mode="json"
        )
    )

    print("\nEVALUATION:")
    print(
        result["evaluation"].model_dump(
            mode="json"
        )
    )

    print("\nREVISED:")
    print(
        result["revised"]
    )

    print("\nFINAL PLAN:")
    print(
        result["final_plan"].model_dump(
            mode="json"
        )
    )


if __name__ == "__main__":
    main()