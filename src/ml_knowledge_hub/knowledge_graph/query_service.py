"""Query services for the knowledge graph."""

"""High-level query service for the ML knowledge graph."""

import json

from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)


class GraphQueryService:
    """
    High-level interface for querying Neo4j.

    This layer sits between:
        KnowledgeAssistant
            ↓
        GraphQueryService
            ↓
        Neo4jStore

    The assistant should not need to know Cypher details.
    """

    def __init__(
        self,
        store: Neo4jStore,
    ):
        self.store = store

    def get_project_models(
        self,
        project_id: str,
    ) -> list[dict]:
        """
        Return models used by a project.
        """

        return self.store.get_models_for_project(
            project_id
        )

    def get_model_datasets(
        self,
        model_id: str,
    ) -> list[dict]:
        """
        Return training and evaluation datasets
        connected to a model.
        """

        return self.store.get_datasets_for_model(
            model_id
        )

    def get_model_metrics(
        self,
        model_id: str,
    ) -> list[dict]:
        """
        Return metrics reported by a model.

        Convert properties_json back into a Python
        dictionary so callers do not need to parse JSON.
        """

        results = self.store.get_metrics_for_model(
            model_id
        )

        cleaned_results = []

        for result in results:
            item = dict(result)

            properties_json = item.pop(
                "properties_json",
                None,
            )

            # Convert Neo4j's stored JSON string back
            # into a normal Python dictionary.
            if properties_json:
                try:
                    item["properties"] = json.loads(
                        properties_json
                    )
                except json.JSONDecodeError:
                    item["properties"] = {}
            else:
                item["properties"] = {}

            cleaned_results.append(
                item
            )

        return cleaned_results

    def get_model_context(
        self,
        model_id: str,
    ) -> dict:
        """
        Return a compact graph context for one model.

        This will later be useful as structured evidence
        for GraphRAG.
        """

        return {
            "model_id": model_id,
            "datasets": self.get_model_datasets(
                model_id
            ),
            "metrics": self.get_model_metrics(
                model_id
            ),
            "projects": self.store.get_projects_for_model(
                model_id
            ),
        }