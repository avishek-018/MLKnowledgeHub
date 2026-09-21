"""Tests for the ML Knowledge Hub MCP server."""

from ml_knowledge_hub.mcp_server.server import search_knowledge


def test_search_knowledge_rejects_empty_query():
    """
    Empty queries should be rejected before the full assistant
    stack is initialized.
    """

    result = search_knowledge("   ")

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_input"
    assert result["message"] == "Query must not be empty."