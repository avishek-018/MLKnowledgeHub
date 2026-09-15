"""Regression tests for semantic filter propagation."""

from ml_knowledge_hub.assistant.service import KnowledgeAssistant
from ml_knowledge_hub.query.router import QueryPlan


class FakeRouter:
    def route(self, query: str) -> QueryPlan:
        return QueryPlan(
            query_type="semantic",
            query=query,
            operation="semantic_search",
            asset_types=["model_card"],
        )


class FakeEmbedder:
    def encode(self, texts):
        return [[0.1, 0.2]]


class RecordingVectorStore:
    def __init__(self):
        self.search_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return []


def test_semantic_plan_asset_types_reach_vector_store():
    vector_store = RecordingVectorStore()
    assistant = KnowledgeAssistant(
        metadata_service=None,
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        query_router=FakeRouter(),
    )

    result = assistant.ask(
        "Which model cards describe AI image detectors?"
    )

    assert vector_store.search_kwargs["asset_types"] == [
        "model_card"
    ]
    assert result["filter"] == {
        "asset_types": ["model_card"]
    }

