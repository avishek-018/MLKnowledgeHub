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


def test_search_knowledge_handles_greeting_without_assistant(monkeypatch):
    def fail_if_initialized():
        raise AssertionError("Heavy assistant should not be initialized")

    monkeypatch.setattr(
        "ml_knowledge_hub.mcp_server.server.get_assistant",
        fail_if_initialized,
    )

    result = search_knowledge("Hello!")

    assert result["status"] == "success"
    assert result["result"]["type"] == "conversation"
    assert result["result"]["operation"] == "greeting"
