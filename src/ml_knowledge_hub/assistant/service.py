"""Assistant service orchestration."""

from ml_knowledge_hub.query.router import route_query
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)

class KnowledgeAssistant:
    def __init__(
        self,
        metadata_service,
        vector_store,
        embedder,
        rag_generator=None,
        graph_service=None,
        entity_registry=None,
    ):
        self.metadata_service = metadata_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.rag_generator = rag_generator
        self.graph_service = graph_service

        # Canonical KG entity registry used to resolve
        # names and aliases mentioned in user questions.
        self.entity_registry = entity_registry

    def ask(self, query: str) -> dict:
        plan = route_query(query)

        # --------------------------------------------------
        # Graph query path
        # --------------------------------------------------
        if plan.query_type == "graph":
            return self._handle_graph_query(
                query=query,
                operation=plan.operation,
            )

        # Structured metadata query
        if plan.query_type == "metadata":
            return self._handle_metadata(plan)

        # Semantic query with metadata filtering
        if plan.query_type == "filtered_semantic":
            return self._handle_semantic(
                query=plan.query,
                asset_types=plan.asset_types,
            )

        # General semantic query
        return self._handle_semantic(
            query=plan.query,
            asset_types=None,
        )

    def _handle_metadata(self, plan) -> dict:
        if plan.operation == "count_projects":
            count = self.metadata_service.count_projects()

            return {
                "type": "metadata",
                "answer": f"We have {count} projects.",
                "data": {
                    "count": count,
                },
            }

        if plan.operation == "list_projects":
            projects = self.metadata_service.list_projects()

            return {
                "type": "metadata",
                "answer": projects,
                "data": {
                    "projects": projects,
                },
            }

        if plan.operation == "count_model_cards":
            model_cards = self.metadata_service.get_assets_by_type(
                "model_card"
            )

            return {
                "type": "metadata",
                "answer": f"We have {len(model_cards)} model cards.",
                "data": {
                    "count": len(model_cards),
                },
            }

        raise ValueError(
            f"Unsupported metadata operation: {plan.operation}"
        )

    def _deduplicate_by_document(self, results):
        seen = set()
        unique_results = []

        for result in results:
            document_id = result.payload["document_id"]

            if document_id in seen:
                continue

            seen.add(document_id)
            unique_results.append(result)

        return unique_results

    def _deduplicate_by_project(self, results):
        seen = set()
        unique_results = []

        for result in results:
            project_id = result.payload["project_id"]

            if project_id in seen:
                continue

            seen.add(project_id)
            unique_results.append(result)

        return unique_results

    def _handle_semantic(
        self,
        query: str,
        asset_types: list[str] | None,
        limit: int = 5,
    ) -> dict:
        query_embedding = self.embedder.encode([query])[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=15,
            asset_types=asset_types,
        )

        query_lower = query.lower()

        if "project" in query_lower:
            results = self._deduplicate_by_project(results)
        else:
            results = self._deduplicate_by_document(results)

        results = results[:limit]

        if self.rag_generator is not None:
            answer = self.rag_generator.generate(
                question=query,
                retrieved_results=results,
            )
        else:
            answer = None

        sources = [
            {
                "source_id": f"S{index}",
                "title": result.payload.get("title"),
                "project_id": result.payload.get("project_id"),
                "asset_type": result.payload.get("asset_type"),
                "document_id": result.payload.get("document_id"),
                "score": result.score,
                "source_url": result.payload.get("source_url"),
            }
            for index, result in enumerate(
                results,
                start=1,
            )
        ]

        return {
            "type": "semantic",
            "query": query,
            "answer": answer,
            "filter": {
                "asset_types": asset_types,
            },
            "sources": sources,
            "results": results,
        }


    def _handle_graph_query(
        self,
        query: str,
        operation: str | None,
    ) -> dict:
        """
        Handle structured graph queries by resolving entity
        names/aliases against the canonical entity registry.
        """

        if self.graph_service is None:
            raise RuntimeError(
                "GraphQueryService is not configured."
            )

        if self.entity_registry is None:
            raise RuntimeError(
                "EntityRegistry is not configured."
            )

        # --------------------------------------------------
        # Project -> Models
        # --------------------------------------------------
        if operation == "project_models":

            project = self.entity_registry.find_entity(
                query,
                entity_type=EntityType.PROJECT,
            )

            if project is None:
                return {
                    "type": "graph",
                    "answer": (
                        "I could not identify the project "
                        "in the knowledge graph."
                    ),
                    "sources": [],
                    "raw": [],
                }

            results = self.graph_service.get_project_models(
                project.entity_id
            )

            return {
                "type": "graph",
                "answer": results,
                "sources": [],
                "raw": results,
            }

        # --------------------------------------------------
        # Model -> Datasets
        # --------------------------------------------------
        if operation == "model_datasets":

            model = self.entity_registry.find_entity(
                query,
                entity_type=EntityType.MODEL,
            )

            if model is None:
                return {
                    "type": "graph",
                    "answer": (
                        "I could not identify the model "
                        "in the knowledge graph."
                    ),
                    "sources": [],
                    "raw": [],
                }

            results = self.graph_service.get_model_datasets(
                model.entity_id
            )

            return {
                "type": "graph",
                "answer": results,
                "sources": [],
                "raw": results,
            }

        # --------------------------------------------------
        # Model -> Metrics
        # --------------------------------------------------
        if operation == "model_metrics":

            model = self.entity_registry.find_entity(
                query,
                entity_type=EntityType.MODEL,
            )

            if model is None:
                return {
                    "type": "graph",
                    "answer": (
                        "I could not identify the model "
                        "in the knowledge graph."
                    ),
                    "sources": [],
                    "raw": [],
                }

            results = self.graph_service.get_model_metrics(
                model.entity_id
            )

            return {
                "type": "graph",
                "answer": results,
                "sources": [],
                "raw": results,
            }

        return {
            "type": "graph",
            "answer": "Unsupported graph operation.",
            "sources": [],
            "raw": [],
        }