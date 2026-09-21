"""Unit tests for Qdrant metadata filters."""

from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


class FakeQueryResult:
    points = []


class RecordingQdrantClient:
    def __init__(self):
        self.query_kwargs = None

    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return FakeQueryResult()


def test_search_builds_asset_type_filter():
    client = RecordingQdrantClient()
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "test"
    store.client = client

    store.search(
        query_embedding=[0.1, 0.2],
        asset_types=["model_card"],
    )

    query_filter = client.query_kwargs["query_filter"]
    condition = query_filter.must[0]

    assert condition.key == "asset_type"
    assert condition.match.any == ["model_card"]

