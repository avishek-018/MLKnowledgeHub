"""Cached access to the existing ML Knowledge Hub services.

This module deliberately contains initialization only. Retrieval, routing, and
answer generation remain in the existing backend services.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ml_knowledge_hub.registry.service import AssetRegistry
from ml_knowledge_hub.assistant.conversation import (
    build_conversation_result,
    classify_local_conversation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def _export_streamlit_secrets() -> None:
    """Expose root-level Streamlit secrets as environment variables.

    Backend services read credentials with os.getenv, so hosted deployments
    that configure Streamlit secrets instead of a .env file need this bridge.
    Values already set in the environment take precedence.
    """

    try:
        secrets = dict(st.secrets)
    except Exception:
        return
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


_export_streamlit_secrets()


@st.cache_resource(show_spinner=False)
def get_registry() -> AssetRegistry:
    """Return the lightweight JSON asset registry."""

    return AssetRegistry(DATA_DIR / "raw" / "manifest.json")


@st.cache_resource(show_spinner=False)
def get_graph_service():
    """Return one shared graph service and Neo4j connection."""

    from ml_knowledge_hub.knowledge_graph.neo4j_store import Neo4jStore
    from ml_knowledge_hub.knowledge_graph.query_service import GraphQueryService

    return GraphQueryService(store=Neo4jStore())


@st.cache_resource(show_spinner=False)
def get_entity_registry():
    """Return the canonical entity registry used by the assistant."""

    from ml_knowledge_hub.knowledge_graph.entity_registry import EntityRegistry

    return EntityRegistry(DATA_DIR / "knowledge_graph" / "entities.json")


@st.cache_resource(show_spinner=False)
def get_query_router():
    """Return the lightweight planner/router without opening data stores."""

    from ml_knowledge_hub.query.router import QueryRouter

    return QueryRouter()


@st.cache_resource(show_spinner=False)
def get_assistant():
    """Build and cache the existing end-to-end assistant stack."""

    from ml_knowledge_hub.assistant.service import KnowledgeAssistant
    from ml_knowledge_hub.metadata.service import MetadataService
    from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore

    registry = get_registry()
    # Open the local store before importing the heavier ML stack. If another
    # process owns Qdrant's lock, fail quickly without loading Transformers.
    vector_store = QdrantStore(storage_path=DATA_DIR / "qdrant")

    from ml_knowledge_hub.embeddings.embedder import Embedder
    from ml_knowledge_hub.rag.generator import RAGGenerator

    return KnowledgeAssistant(
        metadata_service=MetadataService(registry),
        vector_store=vector_store,
        embedder=Embedder(),
        rag_generator=RAGGenerator(),
        graph_service=get_graph_service(),
        entity_registry=get_entity_registry(),
        query_router=get_query_router(),
    )


@st.cache_resource(show_spinner=False)
def get_contextualizer():
    """Return the follow-up question rewriter."""

    from ml_knowledge_hub.assistant.contextualizer import QueryContextualizer

    return QueryContextualizer()


def ask_knowledge(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan first, then initialize retrieval services only when required.

    With chat history, follow-up questions are rewritten into standalone
    questions before planning so references like "it" resolve correctly.
    """

    standalone_query = query
    if history and classify_local_conversation(query) is None:
        standalone_query = get_contextualizer().rewrite(query, history)

    plan = get_query_router().route(standalone_query)
    if plan.query_type == "conversation":
        result = build_conversation_result(plan.operation)
    else:
        result = get_assistant().ask(standalone_query, plan=plan)

    if standalone_query != query:
        result["standalone_query"] = standalone_query
    return result


@st.cache_data(show_spinner=False)
def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON data file and cache it until its argument changes."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_evaluation_results() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the current planning and end-to-end evaluation reports."""

    evaluation_dir = DATA_DIR / "evaluation"
    planning = load_json(evaluation_dir / "query_planning_results.json")
    retrieval = load_json(evaluation_dir / "answer_retrieval_results.json")
    return planning, retrieval
