"""Tests for knowledge graph resolution policies."""

from ml_knowledge_hub.knowledge_graph.resolution_policy import (
    apply_resolution_policy,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    ResolutionAction,
    ResolutionDecision,
)


def test_high_confidence_merge():
    decision = ResolutionDecision(
        action=ResolutionAction.MERGE,
        canonical_entity_id="stable_diffusion_xl",
        confidence=0.99,
        reason="Strong alias match.",
        matched_name="Stable Diffusion XL",
    )

    result = apply_resolution_policy(decision)

    assert result.action == ResolutionAction.MERGE


def test_low_confidence_merge_becomes_review():
    decision = ResolutionDecision(
        action=ResolutionAction.MERGE,
        canonical_entity_id="stable_diffusion_xl",
        confidence=0.80,
        reason="Possible match.",
    )

    result = apply_resolution_policy(decision)

    assert result.action == ResolutionAction.REVIEW


def test_high_confidence_new():
    decision = ResolutionDecision(
        action=ResolutionAction.NEW,
        canonical_entity_id=None,
        confidence=0.96,
        reason="Distinct model version.",
    )

    result = apply_resolution_policy(decision)

    assert result.action == ResolutionAction.NEW


def test_review_stays_review():
    decision = ResolutionDecision(
        action=ResolutionAction.REVIEW,
        canonical_entity_id=None,
        confidence=0.98,
        reason="Conflicting evidence.",
    )

    result = apply_resolution_policy(decision)

    assert result.action == ResolutionAction.REVIEW