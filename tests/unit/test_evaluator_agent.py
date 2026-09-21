"""Test the Evaluator Agent independently."""

from dotenv import load_dotenv

from ml_knowledge_hub.agents.evaluator_agent import (
    EvaluatorAgent,
)
from ml_knowledge_hub.query.planner_schema import (
    PlannedEntityType,
    PlannedOperation,
    PlannedQueryType,
    QueryPlanLLM,
)


def run_test(
    evaluator: EvaluatorAgent,
    question: str,
    plan: QueryPlanLLM,
):
    print()
    print("=" * 100)
    print("QUESTION:")
    print(question)

    print("\nPROPOSED PLAN:")
    print(
        plan.model_dump(
            mode="json"
        )
    )

    evaluation = evaluator.evaluate(
        query=question,
        plan=plan,
    )

    print("\nEVALUATION:")
    print(
        evaluation.model_dump(
            mode="json"
        )
    )


def main():
    load_dotenv()

    evaluator = EvaluatorAgent()

    # --------------------------------------------------
    # Case 1:
    # A good graph plan.
    #
    # The evaluator should normally accept this.
    # --------------------------------------------------
    good_plan = QueryPlanLLM(
        query_type=PlannedQueryType.GRAPH,
        operation=PlannedOperation.MODEL_DATASETS,
        entity_mention="AI-generated images detector",
        entity_type=PlannedEntityType.MODEL,
        target_entity_type=PlannedEntityType.DATASET,
        asset_types=None,
    )

    run_test(
        evaluator=evaluator,
        question=(
            "What data was the AI-generated images detector "
            "evaluated on?"
        ),
        plan=good_plan,
    )

    # --------------------------------------------------
    # Case 2:
    # A deliberately bad semantic plan.
    #
    # The question asks for a direct project -> model
    # relationship, which our graph supports.
    #
    # The evaluator should reject this and recommend:
    # graph / project_models.
    # --------------------------------------------------
    bad_plan = QueryPlanLLM(
        query_type=PlannedQueryType.SEMANTIC,
        operation=PlannedOperation.SEMANTIC_SEARCH,
        entity_mention="NYUAD",
        entity_type=PlannedEntityType.PROJECT,
        target_entity_type=PlannedEntityType.MODEL,
        asset_types=None,
    )

    run_test(
        evaluator=evaluator,
        question="What models belong to NYUAD?",
        plan=bad_plan,
    )


if __name__ == "__main__":
    main()