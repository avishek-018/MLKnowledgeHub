"""Policies governing knowledge graph entity resolution."""

"""Policy for applying entity-resolution decisions."""

from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    ResolutionAction,
    ResolutionDecision,
)


MERGE_THRESHOLD = 0.95
NEW_THRESHOLD = 0.90


def apply_resolution_policy(
    decision: ResolutionDecision,
) -> ResolutionDecision:

    if decision.action == ResolutionAction.MERGE:
        if (
            decision.confidence >= MERGE_THRESHOLD
            and decision.canonical_entity_id is not None
        ):
            return decision

        return ResolutionDecision(
            action=ResolutionAction.REVIEW,
            canonical_entity_id=None,
            confidence=decision.confidence,
            reason=(
                "Merge confidence did not meet "
                "the automatic merge threshold. "
                f"Original reason: {decision.reason}"
            ),
            matched_name=decision.matched_name,
        )

    if decision.action == ResolutionAction.NEW:
        if decision.confidence >= NEW_THRESHOLD:
            return decision

        return ResolutionDecision(
            action=ResolutionAction.REVIEW,
            canonical_entity_id=None,
            confidence=decision.confidence,
            reason=(
                "New-entity confidence did not meet "
                "the automatic creation threshold. "
                f"Original reason: {decision.reason}"
            ),
            matched_name=None,
        )

    return decision