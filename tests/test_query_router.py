from ml_knowledge_hub.query.planner_schema import (
    PlannedAssetType,
    PlannedOperation,
    PlannedQueryType,
    QueryPlanLLM,
)
from ml_knowledge_hub.query.router import (
    QueryRouter,
)


class FakeOrchestrator:
    """Return a deterministic final plan without making LLM calls."""

    def __init__(
        self,
        plan: QueryPlanLLM,
    ):
        self.final_plan = plan

    def create_plan(
        self,
        query: str,
    ) -> dict:
        return {"final_plan": self.final_plan}


def make_router(plan: QueryPlanLLM) -> QueryRouter:
    return QueryRouter(
        orchestrator=FakeOrchestrator(plan)
    )


def test_metadata_count_projects():
    router = make_router(
        QueryPlanLLM(
            query_type=PlannedQueryType.METADATA,
            operation=PlannedOperation.COUNT_PROJECTS,
        )
    )

    plan = router.route(
        "How many projects do we have?"
    )

    assert plan.query_type == "metadata"
    assert plan.operation == "count_projects"


def test_graph_project_models():
    router = make_router(
        QueryPlanLLM(
            query_type=PlannedQueryType.GRAPH,
            operation=PlannedOperation.PROJECT_MODELS,
            entity_mention="NYUAD",
            entity_type="project",
            asset_types=None,
        )
    )

    plan = router.route(
        "Which models are used by NYUAD?"
    )

    assert plan.query_type == "graph"
    assert plan.operation == "project_models"
    assert plan.entity_mention == "NYUAD"


def test_hybrid_model_context():
    router = make_router(
        QueryPlanLLM(
            query_type=PlannedQueryType.HYBRID,
            operation=PlannedOperation.MODEL_CONTEXT,
            entity_mention="AI-generated images detector",
            entity_type="model",
            asset_types=None,
        )
    )

    plan = router.route(
        "Give me details about the AI-generated images detector."
    )

    assert plan.query_type == "hybrid"
    assert plan.operation == "model_context"
    assert (
        plan.entity_mention
        == "AI-generated images detector"
    )


def test_semantic_plan_preserves_asset_types():
    router = make_router(
        QueryPlanLLM(
            query_type=PlannedQueryType.SEMANTIC,
            operation=PlannedOperation.SEMANTIC_SEARCH,
            asset_types=[PlannedAssetType.MODEL_CARD],
        )
    )

    plan = router.route(
        "Which model cards describe AI image detectors?"
    )

    assert plan.query_type == "semantic"
    assert plan.asset_types == ["model_card"]
