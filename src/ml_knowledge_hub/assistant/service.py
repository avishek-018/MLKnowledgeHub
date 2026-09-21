"""Assistant service orchestration."""

from ml_knowledge_hub.query.router import (
    QueryPlan,
    QueryRouter,
)
from ml_knowledge_hub.assistant.conversation import (
    build_conversation_result,
)
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
        query_router=None,
    ):
        self.metadata_service = metadata_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.rag_generator = rag_generator
        self.graph_service = graph_service
        self.entity_registry = entity_registry

        # --------------------------------------------------
        # Query planner/router
        #
        # A router can be injected during tests.
        # Otherwise, use the normal LLM-backed QueryRouter.
        # --------------------------------------------------
        self.query_router = (
            query_router
            if query_router is not None
            else QueryRouter()
        )

    def ask(
        self,
        query: str,
        plan: QueryPlan | None = None,
    ) -> dict:
        # --------------------------------------------------
        # Build a structured execution plan for this question.
        # --------------------------------------------------
        plan = plan or self.query_router.route(query)

        if plan.query_type == "conversation":
            return build_conversation_result(plan.operation)

        # --------------------------------------------------
        # Graph query path
        # --------------------------------------------------
        if plan.query_type == "graph":
            return self._handle_graph_query(
                query=query,
                operation=plan.operation,
                entity_mention=plan.entity_mention,
            )

        if plan.query_type == "hybrid":
            return self._handle_hybrid_query(
                query=query,
                operation=plan.operation,
                entity_mention=plan.entity_mention,
            )

        # Structured metadata query
        if plan.query_type == "metadata":
            return self._handle_metadata(plan)

        # --------------------------------------------------
        # Semantic retrieval.
        #
        # asset_types may be None for broad semantic search
        # or contain one or more artifact types selected by
        # the agentic planning workflow.
        # --------------------------------------------------
        if plan.query_type == "semantic":
            return self._handle_semantic(
                query=plan.query,
                asset_types=plan.asset_types,
            )

        raise ValueError(
            f"Unsupported query type: {plan.query_type}"
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
        entity_mention: str | None = None,
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
            lookup_text = entity_mention or query
            project = self.entity_registry.find_entity(
                lookup_text,
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
            lookup_text = entity_mention or query
            model = self.entity_registry.find_entity(
                lookup_text,
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
            lookup_text = entity_mention or query
            model = self.entity_registry.find_entity(
                lookup_text,
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

    def _handle_hybrid_query(
        self,
        query: str,
        operation: str | None,
        entity_mention: str | None = None,
    ) -> dict:
        """
        Combine Neo4j graph structure with Qdrant semantic
        evidence and synthesize one grounded answer.
        """

        if self.graph_service is None:
            raise RuntimeError(
                "GraphQueryService is not configured."
            )

        if self.entity_registry is None:
            raise RuntimeError(
                "EntityRegistry is not configured."
            )

        if self.rag_generator is None:
            raise RuntimeError(
                "RAG generator is not configured."
            )

        # --------------------------------------------------
        # For the first hybrid operation, resolve a MODEL
        # mentioned in the user's question.
        # --------------------------------------------------
        if operation == "model_context":
            lookup_text = entity_mention or query
            model = self.entity_registry.find_entity(
                lookup_text,
                entity_type=EntityType.MODEL,
            )

            if model is None:
                return {
                    "type": "hybrid",
                    "answer": (
                        "I could not identify the model "
                        "in the knowledge graph."
                    ),
                    "sources": [],
                    "graph_context": {},
                    "raw": [],
                }

            # --------------------------------------------------
            # 1. Retrieve structured graph evidence from Neo4j.
            # --------------------------------------------------
            graph_context = (
                self.graph_service.get_model_context(
                    model.entity_id
                )
            )

            # --------------------------------------------------
            # 2. Retrieve supporting text evidence from Qdrant.
            #
            # IMPORTANT:
            # The graph has already identified which project(s)
            # this model belongs to.
            #
            # Instead of searching the entire corpus and hoping
            # relevant project documents appear in the top-k,
            # use the graph project IDs to constrain Qdrant
            # retrieval directly.
            # --------------------------------------------------
            query_embedding = self.embedder.encode(
                [query]
            )[0]

            project_ids = {
                project["entity_id"]
                for project in graph_context.get(
                    "projects",
                    []
                )
            }

            evidence_results = []


            # --------------------------------------------------
            # Search each graph-linked project independently.
            # --------------------------------------------------
            if project_ids:

                for project_id in project_ids:

                    project_hits = self.vector_store.search(
                        query_embedding=query_embedding,
                        limit=10,
                        project_id=project_id,
                    )

                    evidence_results.extend(
                        project_hits
                    )

            else:
                # --------------------------------------------------
                # If the graph has no project relationship, fall
                # back to ordinary semantic retrieval.
                #
                # This fallback is allowed only when there is no
                # graph-linked project at all.
                # --------------------------------------------------
                evidence_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    limit=10,
                )


            # --------------------------------------------------
            # Remove repeated chunks/documents and keep a small,
            # high-quality evidence set for synthesis.
            # --------------------------------------------------
            evidence_results = (
                self._deduplicate_by_document(
                    evidence_results
                )[:5]
            )

            # --------------------------------------------------
            # 3. Ask the RAG generator to synthesize using BOTH
            # graph facts and document evidence.
            # --------------------------------------------------
            answer = (
                self.rag_generator.generate_hybrid(
                    question=query,
                    graph_context=graph_context,
                    retrieved_results=evidence_results,
                )
            )

            # --------------------------------------------------
            # 4. Build normal source metadata for citations.
            # --------------------------------------------------
            sources = []

            for index, result in enumerate(
                evidence_results,
                start=1,
            ):
                sources.append(
                    {
                        "source_id": f"S{index}",
                        "title": result.payload.get(
                            "title"
                        ),
                        "project_id": result.payload.get(
                            "project_id"
                        ),
                        "asset_type": result.payload.get(
                            "asset_type"
                        ),
                        "document_id": result.payload.get(
                            "document_id"
                        ),
                        "score": result.score,
                        "source_url": result.payload.get(
                            "source_url"
                        ),
                    }
                )

            return {
                "type": "hybrid",
                "answer": answer,
                "sources": sources,
                "graph_context": graph_context,
                "raw": evidence_results,
            }

        # --------------------------------------------------
        # Project-level hybrid context
        # --------------------------------------------------
        if operation == "project_context":
            lookup_text = entity_mention or query

            project = self.entity_registry.find_entity(
                lookup_text,
                entity_type=EntityType.PROJECT,
            )

            if project is None:
                return {
                    "type": "hybrid",
                    "answer": (
                        "I could not identify the project "
                        "in the knowledge graph."
                    ),
                    "sources": [],
                    "graph_context": {},
                    "raw": [],
                }

            # --------------------------------------------------
            # 1. Retrieve structured graph evidence.
            # --------------------------------------------------
            graph_context = (
                self.graph_service.get_project_context(
                    project.entity_id
                )
            )

            # --------------------------------------------------
            # 2. Retrieve textual evidence only from documents
            # belonging to this project.
            # --------------------------------------------------
            query_embedding = self.embedder.encode(
                [query]
            )[0]

            evidence_results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=10,
                project_id=project.entity_id,
            )

            # --------------------------------------------------
            # Deduplicate documents and keep a compact evidence
            # set for answer generation.
            # --------------------------------------------------
            evidence_results = (
                self._deduplicate_by_document(
                    evidence_results
                )[:5]
            )

            # --------------------------------------------------
            # 3. Synthesize graph + document evidence.
            # --------------------------------------------------
            answer = (
                self.rag_generator.generate_hybrid(
                    question=query,
                    graph_context=graph_context,
                    retrieved_results=evidence_results,
                )
            )

            # --------------------------------------------------
            # 4. Build source metadata.
            # --------------------------------------------------
            sources = []

            for index, result in enumerate(
                evidence_results,
                start=1,
            ):
                sources.append(
                    {
                        "source_id": f"S{index}",
                        "title": result.payload.get(
                            "title"
                        ),
                        "project_id": result.payload.get(
                            "project_id"
                        ),
                        "asset_type": result.payload.get(
                            "asset_type"
                        ),
                        "document_id": result.payload.get(
                            "document_id"
                        ),
                        "score": result.score,
                        "source_url": result.payload.get(
                            "source_url"
                        ),
                    }
                )

            return {
                "type": "hybrid",
                "answer": answer,
                "sources": sources,
                "graph_context": graph_context,
                "raw": evidence_results,
            }
        return {
            "type": "hybrid",
            "answer": "Unsupported hybrid operation.",
            "sources": [],
            "graph_context": {},
            "raw": [],
        }
