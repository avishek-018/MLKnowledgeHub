"""Tests for lightweight and scoped conversation behavior."""

import pytest

from ml_knowledge_hub.assistant.conversation import (
    build_conversation_result,
    classify_local_conversation,
)
from ml_knowledge_hub.assistant.service import KnowledgeAssistant
from ml_knowledge_hub.query.router import QueryPlan


@pytest.mark.parametrize(
    ("query", "operation"),
    [
        ("hello", "greeting"),
        ("Good morning!", "greeting"),
        ("What can you do?", "help"),
        ("Thank you", "acknowledgement"),
        ("See you later!", "goodbye"),
    ],
)
def test_classifies_unambiguous_local_conversation(query, operation):
    assert classify_local_conversation(query) == operation


def test_does_not_swallow_knowledge_query_with_greeting():
    assert (
        classify_local_conversation("Hi, what models are used by NYUAD?")
        is None
    )


def test_out_of_scope_result_redirects_to_hub():
    result = build_conversation_result("out_of_scope")

    assert result["type"] == "conversation"
    assert result["operation"] == "out_of_scope"
    assert "registered project" in result["answer"]
    assert result["sources"] == []


def test_assistant_executes_precomputed_conversation_without_dependencies():
    assistant = KnowledgeAssistant(
        metadata_service=None,
        vector_store=None,
        embedder=None,
        query_router=None,
    )
    plan = QueryPlan(
        query_type="conversation",
        query="What is the weather today?",
        operation="out_of_scope",
    )

    result = assistant.ask("What is the weather today?", plan=plan)

    assert result["type"] == "conversation"
    assert result["operation"] == "out_of_scope"
