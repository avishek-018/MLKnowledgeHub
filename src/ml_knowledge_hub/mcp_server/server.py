"""
MCP server for ML Knowledge Hub.

Lightweight MCP tools are available immediately.
The full KnowledgeAssistant stack is initialized lazily
only when search_knowledge() is called for the first time.
"""

from dotenv import load_dotenv
from mcp.server import MCPServer

from ml_knowledge_hub.registry.service import AssetRegistry
import time
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
# Load API keys / Neo4j credentials before any service needs them.
load_dotenv()


# ------------------------------------------------------------------
# MCP server
# ------------------------------------------------------------------
mcp = MCPServer("ML Knowledge Hub")


# ------------------------------------------------------------------
# Lightweight registry
#
# Safe to initialize immediately because it does not load ML models,
# connect to Neo4j, or open the vector store.
# ------------------------------------------------------------------
registry = AssetRegistry(
    registry_path="data/raw/manifest.json"
)


# ------------------------------------------------------------------
# Lazy assistant singleton
#
# None means the expensive assistant stack has not been loaded yet.
# ------------------------------------------------------------------
_assistant = None
_query_router = None


def get_query_router():
    """Return a shared router without initializing retrieval services."""

    global _query_router

    if _query_router is None:
        from ml_knowledge_hub.query.router import QueryRouter

        _query_router = QueryRouter()

    return _query_router


def get_assistant():
    """
    Build the complete KnowledgeAssistant only when first needed.

    Subsequent MCP calls reuse the same initialized assistant.
    """

    global _assistant

    # Reuse the already-created assistant after the first call.
    if _assistant is not None:
        return _assistant

    # --------------------------------------------------------------
    # Import expensive components lazily.
    #
    # This keeps MCP startup fast and prevents model initialization
    # from delaying the initial MCP handshake.
    # --------------------------------------------------------------
    from ml_knowledge_hub.metadata.service import MetadataService
    from ml_knowledge_hub.embeddings.embedder import Embedder
    from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore
    from ml_knowledge_hub.rag.generator import RAGGenerator
    from ml_knowledge_hub.knowledge_graph.neo4j_store import Neo4jStore
    from ml_knowledge_hub.knowledge_graph.query_service import GraphQueryService
    from ml_knowledge_hub.knowledge_graph.entity_registry import EntityRegistry
    from ml_knowledge_hub.assistant.service import KnowledgeAssistant

    # --------------------------------------------------------------
    # Existing ML Knowledge Hub services
    # --------------------------------------------------------------
    metadata_service = MetadataService(
        registry
    )

    embedder = Embedder()

    vector_store = QdrantStore()

    rag_generator = RAGGenerator()

    neo4j_store = Neo4jStore()

    graph_service = GraphQueryService(
        store=neo4j_store
    )

    entity_registry = EntityRegistry(
        "data/knowledge_graph/entities.json"
    )

    # --------------------------------------------------------------
    # Full agentic assistant
    # --------------------------------------------------------------
    _assistant = KnowledgeAssistant(
        metadata_service=metadata_service,
        vector_store=vector_store,
        embedder=embedder,
        rag_generator=rag_generator,
        graph_service=graph_service,
        entity_registry=entity_registry,
        query_router=get_query_router(),
    )

    return _assistant


# ------------------------------------------------------------------
# MCP tools
# ------------------------------------------------------------------
@mcp.tool()
def health_check() -> dict:
    """
    Verify that ML Knowledge Hub MCP is running.
    """

    return {
        "status": "ok",
        "service": "ML Knowledge Hub",
    }


@mcp.tool()
def list_projects() -> list[str]:
    """
    Return all registered ML project IDs.
    """

    return registry.list_projects()


@mcp.tool()
def count_projects() -> dict:
    """
    Return the number of registered ML projects.
    """

    return {
        "count": registry.count_projects(),
    }


@mcp.tool()
def search_knowledge(query: str) -> dict:
    if not query or not query.strip():
        return {
            "status": "error",
            "error_type": "invalid_input",
            "message": "Query must not be empty.",
        }

    try:
        start = time.perf_counter()
        clean_query = query.strip()
        plan = get_query_router().route(clean_query)

        if plan.query_type == "conversation":
            from ml_knowledge_hub.assistant.conversation import (
                build_conversation_result,
            )

            result = build_conversation_result(plan.operation)
        else:
            assistant = get_assistant()
            result = assistant.ask(clean_query, plan=plan)
        elapsed = time.perf_counter() - start
        

        return {
            "latency_seconds": round(elapsed, 2),
            "status": "success",
            "result": result,
        }

    except Exception:
        # Keep the detailed traceback in server logs for debugging,
        # but do not expose internal implementation details to MCP clients.
        logger.exception("search_knowledge failed")

        return {
            "status": "error",
            "error_type": "internal_error",
            "message": "The knowledge search could not be completed.",
        }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
