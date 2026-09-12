"""Qdrant vector store integration."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny


class QdrantStore:
    def __init__(
        self,
        collection_name: str = "ml_assets",
        vector_size: int = 384,
    ):
        self.collection_name = collection_name

        # Local in-memory Qdrant for development/testing
        self.client = QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def add_chunks(
        self,
        chunks,
        embeddings,
    ):
        points = []

        for idx, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "project_id": chunk.project_id,
                        "asset_type": chunk.asset_type,
                        "title": chunk.title,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "source_url": chunk.source_url,
                        "license": chunk.license,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # def search(
    #     self,
    #     query_embedding,
    #     limit: int = 5,
    # ):
    #     results = self.client.query_points(
    #         collection_name=self.collection_name,
    #         query=query_embedding,
    #         limit=limit,
    #     )

    #     return results.points




    def search(
        self,
        query_embedding,
        limit: int = 5,
        asset_types: list[str] | None = None,
        project_id: str | None = None,
    ):
        must_conditions = []

        # Optional asset-type filter
        if asset_types:
            must_conditions.append(
                FieldCondition(
                    key="asset_type",
                    match=MatchAny(any=asset_types),
                )
            )

        # Optional project filter
        if project_id is not None:
            must_conditions.append(
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            )

        query_filter = (
            Filter(must=must_conditions)
            if must_conditions
            else None
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        return results.points