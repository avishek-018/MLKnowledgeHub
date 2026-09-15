"""Test the Revision Agent independently."""

from dotenv import load_dotenv

from ml_knowledge_hub.agents.revision_agent import (
    RevisionAgent,
)
from ml_knowledge_hub.agents.schemas import (
    PlanEvaluation,
)
from ml_knowledge_hub.query.planner_schema import (
    PlannedEntityType,
    PlannedOperation,
    PlannedQueryType,
    QueryPlanLLM,
)


def main():
    load_dotenv()

    revision_agent = RevisionAgent()

    # --------------------------------------------------
    # Original user question.
    # --------------------------------------------------
    question = "What models belong to NYUAD?"

    # --------------------------------------------------
    # Deliberately incorrect initial plan.
    # --------------------------------------------------
    original_plan = QueryPlanLLM(
        query_type=PlannedQueryType.SEMANTIC,
        operation=PlannedOperation.SEMANTIC_SEARCH,
        entity_mention="NYUAD",
        entity_type=PlannedEntityType.PROJECT,
        target_entity_type=PlannedEntityType.MODEL,
        asset_types=None,
    )

    # --------------------------------------------------
    # Feedback similar to what our Evaluator Agent
    # already produced.
    # --------------------------------------------------
    evaluation = PlanEvaluation(
        valid=False,
        feedback=(
            "The plan uses semantic search for a direct "
            "project-to-model relationship that is available "
            "through the graph."
        ),
        issues=[
            (
                "The selected query type and operation are not "
                "the most precise capabilities for identifying "
                "models associated with a project."
            )
        ],
        suggested_query_type=PlannedQueryType.GRAPH,
        suggested_operation=PlannedOperation.PROJECT_MODELS,
        suggested_entity_type=None,
        suggested_target_entity_type=None,
        suggested_asset_types=None,
    )

    print("=" * 100)
    print("QUESTION:")
    print(question)

    print("\nORIGINAL PLAN:")
    print(
        original_plan.model_dump(
            mode="json"
        )
    )

    print("\nEVALUATOR FEEDBACK:")
    print(
        evaluation.model_dump(
            mode="json"
        )
    )

    revised_plan = revision_agent.revise(
        query=question,
        original_plan=original_plan,
        evaluation=evaluation,
    )

    print("\nREVISED PLAN:")
    print(
        revised_plan.model_dump(
            mode="json"
        )
    )


if __name__ == "__main__":
    main()