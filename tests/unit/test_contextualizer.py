"""Tests for follow-up question rewriting."""

from types import SimpleNamespace

from ml_knowledge_hub.assistant.contextualizer import (
    QueryContextualizer,
    format_history,
)


class FakeResponses:
    def __init__(self, output_text="", error=None):
        self.output_text = output_text
        self.error = error
        self.prompts = []

    def create(self, model, input):
        self.prompts.append(input)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


def make_contextualizer(responses: FakeResponses) -> QueryContextualizer:
    contextualizer = QueryContextualizer.__new__(QueryContextualizer)
    contextualizer.client = SimpleNamespace(responses=responses)
    contextualizer.model = "test-model"
    return contextualizer


HISTORY = [
    {"role": "user", "content": "What do we know about GenImage?"},
    {
        "role": "assistant",
        "content": "GenImage is a benchmark for AI-generated image detection.",
        "result": {"answer": "GenImage is a benchmark for AI-generated image detection."},
    },
]


def test_follow_up_is_rewritten_with_history():
    responses = FakeResponses('"Which datasets does GenImage use?"')

    rewritten = make_contextualizer(responses).rewrite(
        "Which datasets does it use?", HISTORY
    )

    assert rewritten == "Which datasets does GenImage use?"
    assert "User: What do we know about GenImage?" in responses.prompts[0]
    assert "Which datasets does it use?" in responses.prompts[0]


def test_no_history_skips_llm_call():
    responses = FakeResponses("should not be used")

    rewritten = make_contextualizer(responses).rewrite("List projects", [])

    assert rewritten == "List projects"
    assert responses.prompts == []


def test_failure_falls_back_to_original_query():
    responses = FakeResponses(error=RuntimeError("API down"))

    rewritten = make_contextualizer(responses).rewrite("And its metrics?", HISTORY)

    assert rewritten == "And its metrics?"


def test_history_formats_structured_answers_and_skips_errors():
    messages = [
        {"role": "user", "content": "List projects"},
        {
            "role": "assistant",
            "result": {"answer": [{"name": "GenImage"}, {"entity_id": "dire"}]},
        },
        {"role": "user", "content": "Broken question"},
        {"role": "assistant", "content": "Search failed", "error": True},
    ]

    transcript = format_history(messages)

    assert "Assistant: GenImage, dire" in transcript
    assert "Search failed" not in transcript
