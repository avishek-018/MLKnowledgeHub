"""Tests for UI response normalization helpers."""

from ml_knowledge_hub.ui.components import to_jsonable
from ml_knowledge_hub.ui.pages.ask import user_error_message


class BackendValue:
    def __init__(self):
        self.name = "result"
        self.vector = [0.1, 0.2]


def test_to_jsonable_removes_sensitive_and_vector_fields():
    value = {
        "answer": "Grounded response",
        "password": "not-for-display",
        "nested": {
            "local_path": "/private/data.json",
            "score": 0.91,
        },
    }

    assert to_jsonable(value) == {
        "answer": "Grounded response",
        "nested": {"score": 0.91},
    }


def test_to_jsonable_normalizes_backend_objects():
    assert to_jsonable(BackendValue()) == {"name": "result"}


def test_qdrant_lock_error_has_safe_actionable_message():
    error = RuntimeError(
        "Storage folder /private/index is already accessed by another instance "
        "of Qdrant client."
    )

    message = user_error_message(error)

    assert "Stop the MCP server" in message
    assert "/private/index" not in message
