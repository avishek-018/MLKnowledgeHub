"""Reusable Streamlit components for ML Knowledge Hub."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st


SENSITIVE_KEYS = {
    "api_key",
    "credentials",
    "embedding",
    "local_path",
    "password",
    "secret",
    "token",
    "vector",
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container {max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem;}
          h1, h2, h3 {letter-spacing: -0.025em;}
          [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.035);
          }
          [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.15);}
          .hub-kicker {font-size: .82rem; font-weight: 700; letter-spacing: .11em;
            text-transform: uppercase; color: #64748b; margin-bottom: .35rem;}
          .hub-subtitle {font-size: 1.12rem; color: #64748b; margin: -.55rem 0 1.6rem;}
          .source-meta {color: #64748b; font-size: .9rem;}
          .relation-arrow {color: #64748b; text-align: center; font-size: 1.35rem; padding-top: 1.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f'<div class="hub-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="hub-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _clean_key(key: Any) -> str:
    return str(key).strip().lower()


def to_jsonable(value: Any) -> Any:
    """Convert backend values to safe, JSON-compatible debug output."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
            if _clean_key(key) not in SENSITIVE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(exclude=SENSITIVE_KEYS))
    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict(exclude=SENSITIVE_KEYS))
        except TypeError:
            return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def render_structured(value: Any) -> None:
    """Render strings naturally and structured values readably."""

    if value is None:
        st.info("No generated answer was returned.")
    elif isinstance(value, str):
        st.markdown(value)
    elif isinstance(value, list):
        if not value:
            st.info("No matching graph records were found.")
        elif all(isinstance(item, Mapping) for item in value):
            st.dataframe(to_jsonable(value), width="stretch", hide_index=True)
        else:
            for item in value:
                st.markdown(f"- {item}")
    elif isinstance(value, Mapping):
        st.json(to_jsonable(value))
    else:
        st.write(value)


def render_sources(
    sources: list[dict[str, Any]],
    *,
    show_heading: bool = True,
) -> None:
    if show_heading:
        st.subheader("Sources")
    if not sources:
        st.caption("This result did not include textual sources.")
        return

    for index, source in enumerate(sources, start=1):
        source_id = source.get("source_id") or f"S{index}"
        title = source.get("title") or "Untitled source"
        with st.container(border=True):
            st.markdown(f"**[{source_id}] {title}**")
            details = []
            if source.get("project_id"):
                details.append(f"Project: `{source['project_id']}`")
            if source.get("asset_type"):
                details.append(f"Type: `{source['asset_type']}`")
            score = source.get("score")
            if isinstance(score, (int, float)):
                details.append(f"Score: `{score:.3f}`")
            if source.get("document_id"):
                details.append(f"Document: `{source['document_id']}`")
            if details:
                st.markdown(" · ".join(details))
            if source.get("source_url"):
                st.link_button("View source", source["source_url"])


def render_graph_context(context: dict[str, Any]) -> None:
    if not context:
        return
    st.subheader("Graph context")
    for key, value in context.items():
        label = key.replace("_", " ").title()
        with st.expander(label, expanded=key == "models"):
            render_structured(value)


def render_execution_details(result: dict[str, Any]) -> None:
    sources = result.get("sources") or []
    graph_context = result.get("graph_context")
    details = {
        "Result type": result.get("type", "unknown"),
        "Route": result.get("route") or result.get("type", "unknown"),
        "Operation": result.get("operation") or "Not returned by backend",
        "Retrieved sources": len(sources),
        "Graph context": "Available" if graph_context else "Not included",
    }
    if result.get("filter") is not None:
        details["Filter"] = result["filter"]
    if result.get("target_entity") is not None:
        details["Target entity"] = result["target_entity"]

    with st.expander("Execution details"):
        st.json(to_jsonable(details))


def render_result(result: Any) -> None:
    if not isinstance(result, Mapping):
        st.error("The assistant returned a malformed response.")
        return

    result = dict(result)
    result_type = str(result.get("type", "unknown")).title()
    st.caption(f"Result type · {result_type}")
    st.subheader("Answer")
    render_structured(result.get("answer"))

    if result.get("type") == "conversation":
        return

    render_graph_context(result.get("graph_context") or {})
    render_sources(result.get("sources") or [])
    render_execution_details(result)

    with st.expander("Show raw response"):
        st.json(to_jsonable(result))


def render_chat_result(result: Any) -> None:
    """Render one assistant result compactly inside a chat message."""

    if not isinstance(result, Mapping):
        st.error("The assistant returned a malformed response.")
        return

    result = dict(result)
    render_structured(result.get("answer"))

    if result.get("type") == "conversation":
        return

    sources = result.get("sources") or []
    if sources:
        with st.expander(
            f"Sources ({len(sources)})",
            icon=":material/library_books:",
        ):
            render_sources(sources, show_heading=False)

    graph_context = result.get("graph_context") or {}
    if graph_context:
        with st.expander(
            "Graph context",
            icon=":material/account_tree:",
        ):
            for key, value in graph_context.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                render_structured(value)

    with st.expander(
        "Execution details",
        icon=":material/tune:",
    ):
        details = {
            "Result type": result.get("type", "unknown"),
            "Route": result.get("route") or result.get("type", "unknown"),
            "Operation": result.get("operation") or "Not returned by backend",
            "Retrieved sources": len(sources),
            "Graph context": "Available" if graph_context else "Not included",
        }
        if result.get("filter") is not None:
            details["Filter"] = result["filter"]
        st.json(to_jsonable(details))
