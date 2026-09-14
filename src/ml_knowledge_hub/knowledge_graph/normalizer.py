"""Normalize extracted knowledge graph data."""

import re


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for deterministic comparison.
    """
    value = name.strip().lower()

    # Normalize common version notation.
    value = re.sub(
        r"\bv\s*(\d)",
        r"\1",
        value,
    )

    # Replace punctuation/separators with spaces.
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    # Collapse repeated whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def normalize_entity_id(name: str) -> str:
    normalized = normalize_entity_name(name)

    return normalized.replace(" ", "_")