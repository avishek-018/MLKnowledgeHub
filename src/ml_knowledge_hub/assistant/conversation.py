"""Lightweight responses for social and out-of-scope conversation."""

from __future__ import annotations

import re


CONVERSATION_RESPONSES = {
    "greeting": (
        "Hi! I can help you explore ML projects, models, datasets, evaluations, "
        "repositories, and technical artifacts. What would you like to find?"
    ),
    "help": (
        "I search the knowledge stored in ML Knowledge Hub. You can ask me to "
        "list projects, find models or datasets, summarize a project, locate "
        "implementation resources, or retrieve evaluation evidence."
    ),
    "acknowledgement": (
        "You're welcome! Ask me anytime about the projects and artifacts in ML "
        "Knowledge Hub."
    ),
    "goodbye": "Goodbye! Come back whenever you want to explore the knowledge hub.",
    "out_of_scope": (
        "I'm focused on knowledge stored in ML Knowledge Hub, so I can't help "
        "with that request. Try asking about a registered project, model, dataset, "
        "evaluation, repository, or technical artifact."
    ),
}


LOCAL_CONVERSATION_PHRASES = {
    "greeting": {
        "hi",
        "hello",
        "hey",
        "hello there",
        "hey there",
        "hello hi",
        "hi hello",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    },
    "help": {
        "help",
        "what can you do",
        "how can you help",
        "what do you do",
        "show me what you can do",
        "who are you",
        "can you help me",
    },
    "acknowledgement": {
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you very much",
        "thx",
        "got it",
        "okay thanks",
        "great thanks",
        "appreciate it",
    },
    "goodbye": {
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "take care",
    },
}


def _normalize(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(normalized.split())


def classify_local_conversation(query: str) -> str | None:
    """Classify only unambiguous, standalone social messages.

    Exact matching is intentional: a mixed query such as "Hi, which models are
    used by NYUAD?" must continue through the normal knowledge planner.
    """

    normalized = _normalize(query)
    for operation, phrases in LOCAL_CONVERSATION_PHRASES.items():
        if normalized in phrases:
            return operation
    return None


def build_conversation_result(operation: str | None) -> dict:
    """Build the standard assistant response for a conversation operation."""

    resolved_operation = (
        operation if operation in CONVERSATION_RESPONSES else "out_of_scope"
    )
    return {
        "type": "conversation",
        "operation": resolved_operation,
        "answer": CONVERSATION_RESPONSES[resolved_operation],
        "sources": [],
    }
