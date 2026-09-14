"""Canonical identifier utilities for knowledge graph entities."""

"""Canonical ID generation for resolved KG entities."""

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.normalizer import (
    normalize_entity_id,
    normalize_entity_name,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
)


def _same_entity_identity(
    existing: EntityCandidate,
    candidate: EntityCandidate,
) -> bool:
    """
    Check whether two entities are clearly the same identity
    based on type and deterministic name normalization.

    This is intentionally conservative. Semantic matching is
    already handled earlier by EntityResolver.
    """

    return (
        existing.entity_type == candidate.entity_type
        and normalize_entity_name(existing.name)
        == normalize_entity_name(candidate.name)
    )


def canonical_id_for_new_entity(
    candidate: EntityCandidate,
    registry: EntityRegistry,
) -> str:
    """
    Generate a deterministic, collision-safe canonical ID.

    Strategy
    --------
    PROJECT:
        Preserve the authoritative project ID.

    Other entity types:
        Prefer a normalized name.

    If that ID is already occupied by a DIFFERENT entity,
    use a type-qualified ID instead.

    Example:

        project: diffusion_detection
        model:   diffusion_detection

    becomes:

        project ID -> diffusion_detection
        model ID   -> model__diffusion_detection
    """

    # --------------------------------------------------
    # 1. Choose the preferred ID.
    # --------------------------------------------------
    if candidate.entity_type == EntityType.PROJECT:
        # Project IDs originate from the authoritative
        # asset registry, so never regenerate them.
        preferred_id = candidate.entity_id
    else:
        preferred_id = normalize_entity_id(
            candidate.name
        )

    # --------------------------------------------------
    # 2. Use the preferred ID if it is free.
    # --------------------------------------------------
    existing = registry.get_entity(
        preferred_id
    )

    if existing is None:
        return preferred_id

    # --------------------------------------------------
    # 3. If the preferred ID already represents this
    #    exact identity, reuse it.
    # --------------------------------------------------
    if _same_entity_identity(
        existing,
        candidate,
    ):
        return preferred_id

    # --------------------------------------------------
    # 4. The preferred ID belongs to another entity.
    #
    # Add the entity type to make the identifier unique.
    #
    # Example:
    #   resnet50
    # becomes:
    #   model__resnet50
    # --------------------------------------------------
    name_part = normalize_entity_id(
        candidate.name
    )

    typed_id = (
        f"{candidate.entity_type.value}"
        f"__{name_part}"
    )

    typed_existing = registry.get_entity(
        typed_id
    )

    if typed_existing is None:
        return typed_id

    # --------------------------------------------------
    # 5. Reuse an already-created type-qualified ID
    #    when it represents the same entity.
    # --------------------------------------------------
    if _same_entity_identity(
        typed_existing,
        candidate,
    ):
        return typed_id

    # --------------------------------------------------
    # 6. Rare final fallback.
    #
    # Include the extractor-provided ID as additional
    # deterministic disambiguation.
    # --------------------------------------------------
    raw_id_part = normalize_entity_id(
        candidate.entity_id
    )

    fallback_id = (
        f"{candidate.entity_type.value}"
        f"__{name_part}"
        f"__{raw_id_part}"
    )

    fallback_existing = registry.get_entity(
        fallback_id
    )

    if fallback_existing is None:
        return fallback_id

    if _same_entity_identity(
        fallback_existing,
        candidate,
    ):
        return fallback_id

    raise ValueError(
        "Unable to generate a unique canonical ID for "
        f"{candidate.name!r} "
        f"({candidate.entity_type.value})."
    )