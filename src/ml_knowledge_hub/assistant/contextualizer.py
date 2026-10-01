"""Rewrite follow-up questions into standalone questions.

The planner and retrieval stack handle one self-contained question at a
time. For multi-turn chat, a follow-up such as "which datasets did it use?"
is first rewritten with the recent conversation into "Which datasets did
GenImage use?" so the rest of the pipeline needs no conversation awareness.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI


logger = logging.getLogger(__name__)

# Recent turns are enough to resolve references; older context adds cost
# and invites the rewriter to drag in unrelated topics.
MAX_HISTORY_MESSAGES = 6
MAX_ANSWER_CHARS = 600


def _answer_text(answer: Any) -> str:
    """Return a short plain-text form of a stored assistant answer."""

    if answer is None:
        return ""
    if isinstance(answer, str):
        text = answer
    elif isinstance(answer, list) and all(isinstance(item, dict) for item in answer):
        names = [
            str(item.get("name") or item.get("entity_id") or item.get("project_id"))
            for item in answer
        ]
        text = ", ".join(name for name in names if name and name != "None")
    else:
        text = json.dumps(answer, default=str)
    return text[:MAX_ANSWER_CHARS]


def format_history(messages: list[dict[str, Any]]) -> str:
    """Format chat messages as a compact transcript for the rewriter."""

    lines = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        if message.get("error"):
            continue
        if message.get("role") == "user":
            lines.append(f"User: {message.get('content', '')}")
        else:
            result = message.get("result") or {}
            text = _answer_text(result.get("answer", message.get("content")))
            if text:
                lines.append(f"Assistant: {text}")
    return "\n".join(lines)


class QueryContextualizer:
    def __init__(self, model: str | None = None):
        self.client = OpenAI()
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini",
        )

    def rewrite(self, query: str, history: list[dict[str, Any]]) -> str:
        """
        Return a standalone version of the query.

        Returns the query unchanged when there is no usable history or the
        rewrite fails, so conversation support never blocks answering.
        """

        transcript = format_history(history)
        if not transcript:
            return query

        prompt = f"""
You rewrite follow-up questions for an ML knowledge-base search system.

Given the conversation and the latest user message, rewrite the latest
message into a single standalone question that can be understood without
the conversation.

Rules:
1. Replace pronouns and references such as "it", "that project", "those
   models", or "the second one" with the specific names they refer to.
2. A short follow-up that only names a new subject, such as "what about
   DIRE?" or "and for SynthBuster?", repeats the previous question about
   that new subject. Example: after "Which models does GenImage use?",
   "What about DIRE?" becomes "Which models does DIRE use?".
3. If the latest message is already standalone or starts a new topic,
   return it unchanged. Questions about the knowledge base as a whole,
   such as "How many projects do we have?", are standalone.
4. The search system answers questions about the knowledge base, never
   about this conversation. Never turn a message into a question about
   the conversation itself.
5. Do not answer the question and do not add facts or constraints that
   the user did not ask for.
6. Keep the user's language and wording where possible.
7. Return only the rewritten question, with no quotes or explanation.

Conversation:
{transcript}

Latest user message:
{query}
""".strip()

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            rewritten = response.output_text.strip().strip('"').strip()
        except Exception:
            logger.exception("Query contextualization failed; using original query")
            return query

        return rewritten or query
