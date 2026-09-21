"""Conversational Ask Knowledge Hub page."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from ml_knowledge_hub.ui.components import (
    page_header,
    render_chat_result,
    to_jsonable,
)
from ml_knowledge_hub.ui.services import ask_knowledge


logger = logging.getLogger(__name__)

EXAMPLE_QUERIES = {
    ":material/folder_open: List projects": "How many projects do we have?",
    ":material/account_tree: Explore models": (
        "Which models are used by the NYUAD project?"
    ),
    ":material/summarize: Summarize a project": (
        "What do we know about GenImage as a project?"
    ),
    ":material/code: Find implementation": (
        "What implementation information is available for DIRE?"
    ),
}


def user_error_message(error: Exception) -> str:
    """Return a safe, actionable message for known service failures."""

    if "already accessed by another instance of Qdrant client" in str(error):
        return (
            "The local Qdrant index is already in use. Stop the MCP server or "
            "other Knowledge Hub process, then try again."
        )
    return (
        "The knowledge search could not be completed. Check the configured "
        "services and try again."
    )


def _initialize_chat_state() -> None:
    st.session_state.setdefault("chat_messages", [])


def _clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.pop("chat_suggestion", None)


def _render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    avatar = ":material/person:" if role == "user" else ":material/hub:"
    with st.chat_message(role, avatar=avatar):
        if message.get("error"):
            st.error(message.get("content", "The request could not be completed."))
        elif role == "assistant" and message.get("result") is not None:
            render_chat_result(message["result"])
        else:
            st.markdown(message.get("content", ""))


def _answer_prompt(prompt: str, history) -> None:
    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_messages.append(user_message)
    with history:
        _render_message(user_message)

    with history.chat_message("assistant", avatar=":material/hub:"):
        try:
            with st.status(
                ":shimmer[Understanding your question]",
                type="compact",
            ) as status:
                result = ask_knowledge(prompt)
                status.update(
                    label=(
                        "Response ready"
                        if result.get("type") == "conversation"
                        else "Knowledge retrieved"
                    ),
                    state="complete",
                )
            render_chat_result(result)

            stored_result = to_jsonable(result)
            stored_result.pop("raw", None)
            stored_result.pop("results", None)
            assistant_message = {
                "role": "assistant",
                "content": result.get("answer", ""),
                "result": stored_result,
            }
        except Exception as error:
            message = user_error_message(error)
            if "already in use" in message:
                logger.warning("Knowledge Hub query blocked by the local Qdrant lock")
            else:
                logger.exception("Knowledge Hub query failed")

            st.error(message)
            assistant_message = {
                "role": "assistant",
                "content": message,
                "error": True,
            }

    st.session_state.chat_messages.append(assistant_message)


def render() -> None:
    _initialize_chat_state()

    heading, actions = st.columns([6, 1], vertical_alignment="bottom")
    with heading:
        page_header(
            "Ask Knowledge Hub",
            "Chat with your ML project knowledge, grounded in registered sources.",
            "Agentic GraphRAG",
        )
    with actions:
        st.button(
            "New chat",
            icon=":material/edit_square:",
            key="clear_chat",
            on_click=_clear_chat,
            width="stretch",
            help="Start a new conversation",
        )

    history = st.container(
        height=480,
        border=False,
        key="chat_history",
        autoscroll=True,
    )

    selected = None
    if st.session_state.chat_messages:
        with history:
            for message in st.session_state.chat_messages:
                _render_message(message)
    else:
        with history:
            with st.chat_message("assistant", avatar=":material/hub:"):
                st.markdown(
                    "Hi! Ask me about ML projects, models, datasets, evaluations, "
                    "repositories, or technical artifacts."
                )

        selected = st.pills(
            "Try asking",
            list(EXAMPLE_QUERIES),
            key="chat_suggestion",
            label_visibility="collapsed",
        )

    prompt = st.chat_input(
        "Message ML Knowledge Hub",
        key="chat_prompt",
        submit_mode="disable",
        max_chars=2_000,
    )

    if not st.session_state.chat_messages and selected:
        prompt = EXAMPLE_QUERIES[selected]

    if prompt and prompt.strip():
        _answer_prompt(prompt.strip(), history)
