"""Pipeline orchestration for knowledge graph entity resolution."""

"""End-to-end entity resolution pipeline."""

from datetime import datetime, timezone

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.entity_resolver import (
    EntityResolver,
)
from ml_knowledge_hub.knowledge_graph.normalizer import (
    normalize_entity_id,
)
from ml_knowledge_hub.knowledge_graph.resolution_policy import (
    apply_resolution_policy,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
    ResolutionAction,
    ResolutionDecision,
)

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.canonical_ids import (
    canonical_id_for_new_entity,
)



class EntityResolutionPipeline:
    """
    Resolve extracted entities into canonical entities.

    Possible outcomes:
    - MERGE: reuse an existing canonical entity
    - NEW: create a new canonical entity
    - REVIEW: leave unresolved for manual/future review
    """

    def __init__(
        self,
        registry: EntityRegistry,
        resolver: EntityResolver,
    ):
        self.registry = registry
        self.resolver = resolver

    def resolve_entity(
        self,
        candidate: EntityCandidate,
    ) -> ResolutionDecision:
        """
        Resolve a single extracted entity against the canonical registry.
        """

        # Load all canonical entities currently known to the system.
        existing_entities = self.registry.list_entities()

        # Ask the resolver to determine whether this entity should
        # be merged, created as new, or flagged for review.
        raw_decision = self.resolver.resolve(
            candidate=candidate,
            existing_entities=existing_entities,
        )

        # Apply deterministic confidence thresholds and safety rules.
        final_decision = apply_resolution_policy(
            raw_decision
        )

        # Apply the final decision to the canonical registry.
        self._apply_decision(
            candidate=candidate,
            decision=final_decision,
        )

        # Record the resolution decision for provenance/debugging.
        self._record_history(
            candidate=candidate,
            decision=final_decision,
        )

        return final_decision

    def _apply_decision(
        self,
        candidate: EntityCandidate,
        decision: ResolutionDecision,
    ) -> None:
        """
        Apply a resolved decision to the canonical entity registry.
        """

        # --------------------------------------------------
        # MERGE
        # --------------------------------------------------
        if decision.action == ResolutionAction.MERGE:
            canonical_id = decision.canonical_entity_id

            # The policy should guarantee this for MERGE,
            # but we validate it again defensively.
            if canonical_id is None:
                raise ValueError(
                    "MERGE decision is missing canonical_entity_id."
                )

            canonical_entity = self.registry.get_entity(
                canonical_id
            )

            if canonical_entity is None:
                raise KeyError(
                    f"Canonical entity not found: {canonical_id}"
                )

            # Preserve the new surface form as an alias so the
            # system can recognize it deterministically next time.
            if (
                candidate.name != canonical_entity.name
                and candidate.name not in canonical_entity.aliases
            ):
                canonical_entity.aliases.append(
                    candidate.name
                )

            # Also preserve aliases discovered on the candidate.
            for alias in candidate.aliases:
                if (
                    alias != canonical_entity.name
                    and alias not in canonical_entity.aliases
                ):
                    canonical_entity.aliases.append(alias)

            self.registry.update_entity(
                canonical_entity
            )

            return

        # --------------------------------------------------
        # NEW
        # --------------------------------------------------
        if decision.action == ResolutionAction.NEW:

            # Generate a deterministic canonical ID while
            # protecting against collisions with entities of
            # other types or identities.
            canonical_id = canonical_id_for_new_entity(
                candidate=candidate,
                registry=self.registry,
            )

            new_entity = EntityCandidate(
                entity_id=canonical_id,
                name=candidate.name,
                entity_type=candidate.entity_type,
                aliases=candidate.aliases,
                description=candidate.description,
                properties=candidate.properties,
            )

            existing = self.registry.get_entity(
                canonical_id
            )

            # Only insert if this canonical entity does not
            # already exist. This keeps repeated ingestion
            # idempotent.
            if existing is None:
                self.registry.add_entity(
                    new_entity
                )

            return

        # --------------------------------------------------
        # REVIEW
        # --------------------------------------------------
        if decision.action == ResolutionAction.REVIEW:
            # REVIEW entities are intentionally NOT inserted into
            # the canonical entity list yet.
            #
            # They are preserved in resolution history so they can
            # later be inspected or resolved manually.
            return

    def _record_history(
        self,
        candidate: EntityCandidate,
        decision: ResolutionDecision,
    ) -> None:
        """
        Store a provenance record for every resolution decision.
        """

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "candidate": candidate.model_dump(
                mode="json"
            ),
            "decision": decision.model_dump(
                mode="json"
            ),
        }

        self.registry.record_resolution(
            record
        )