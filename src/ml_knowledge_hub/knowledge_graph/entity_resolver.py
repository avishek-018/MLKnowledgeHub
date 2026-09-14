"""Resolve and link entities in the knowledge graph."""

"""LLM-assisted entity identity resolution."""

import json
import os

from openai import OpenAI

from ml_knowledge_hub.knowledge_graph.normalizer import (
    normalize_entity_id,
    normalize_entity_name,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
    ResolutionAction,
    ResolutionDecision,
    EntityType
)


class EntityResolver:
    def __init__(self, model: str | None = None):
        self.client = OpenAI()

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def resolve(
        self,
        candidate: EntityCandidate,
        existing_entities: list[EntityCandidate],
    ) -> ResolutionDecision:



        # --------------------------------------------------
        # 0. Exact canonical-ID match
        #
        # If both the entity ID and entity type already match,
        # we know this is the same canonical entity.
        # No LLM call is needed.
        # --------------------------------------------------
        for existing in existing_entities:
            if (
                existing.entity_id == candidate.entity_id
                and existing.entity_type == candidate.entity_type
            ):
                return ResolutionDecision(
                    action=ResolutionAction.MERGE,
                    canonical_entity_id=existing.entity_id,
                    confidence=1.0,
                    reason=(
                        "Exact entity ID and entity type match."
                    ),
                    matched_name=existing.name,
                )


        # --------------------------------------------------
        # 0b. Project IDs are authoritative
        #
        # Project IDs come from manifest.json / AssetRegistry.
        # Therefore, a project should not be semantically merged
        # with another project just because their names are similar.
        #
        # If the project ID is new, create a new canonical project.
        # --------------------------------------------------
        if candidate.entity_type == EntityType.PROJECT:
            return ResolutionDecision(
                action=ResolutionAction.NEW,
                canonical_entity_id=None,
                confidence=1.0,
                reason=(
                    "Project identity is determined by the "
                    "authoritative project_id."
                ),
            )
        """
        Resolve a newly extracted entity against existing canonical entities.
        """

        # --------------------------------------------------
        # IMPORTANT:
        # Only entities of the SAME type can represent the
        # same canonical entity.
        #
        # Example:
        #   project "nyuad_ai_image_detector"
        #       must NEVER merge with
        #   model "NYUAD_AI-generated_images_detector"
        # --------------------------------------------------
        compatible_entities = [
            entity
            for entity in existing_entities
            if entity.entity_type == candidate.entity_type
        ]

        # --------------------------------------------------
        # 1. Exact canonical-name match
        # --------------------------------------------------
        candidate_normalized_id = normalize_entity_id(
            candidate.name
        )

        for existing in compatible_entities:
            existing_normalized_id = normalize_entity_id(
                existing.name
            )

            if candidate_normalized_id == existing_normalized_id:
                return ResolutionDecision(
                    action=ResolutionAction.MERGE,
                    canonical_entity_id=existing.entity_id,
                    confidence=1.0,
                    reason=(
                        "Deterministic normalized names match "
                        "and entity types are identical."
                    ),
                    matched_name=existing.name,
                )

        # --------------------------------------------------
        # 2. Exact alias match
        # --------------------------------------------------
        candidate_normalized_name = normalize_entity_name(
            candidate.name
        )

        for existing in compatible_entities:
            aliases = [
                existing.name,
                *existing.aliases,
            ]

            for alias in aliases:
                if (
                    normalize_entity_name(alias)
                    == candidate_normalized_name
                ):
                    return ResolutionDecision(
                        action=ResolutionAction.MERGE,
                        canonical_entity_id=existing.entity_id,
                        confidence=1.0,
                        reason=(
                            "Candidate exactly matches an "
                            "existing alias with the same "
                            "entity type."
                        ),
                        matched_name=alias,
                    )

        # --------------------------------------------------
        # 3. No compatible existing entities
        # --------------------------------------------------
        if not compatible_entities:
            return ResolutionDecision(
                action=ResolutionAction.NEW,
                canonical_entity_id=None,
                confidence=1.0,
                reason=(
                    "No existing canonical entities of "
                    "the same type are available."
                ),
            )

        # --------------------------------------------------
        # 4. LLM semantic resolution
        # --------------------------------------------------
        return self._resolve_with_llm(
            candidate=candidate,
            existing_entities=compatible_entities,
        )

    def _resolve_with_llm(
        self,
        candidate: EntityCandidate,
        existing_entities: list[EntityCandidate],
    ) -> ResolutionDecision:

        existing_data = [
            entity.model_dump(mode="json")
            for entity in existing_entities
        ]

        candidate_data = candidate.model_dump(
            mode="json"
        )

        prompt = f"""
You are resolving entity identities for a machine-learning knowledge graph.

Your task is to determine whether a newly extracted entity refers to the same
real-world entity as one of the existing canonical entities.

Candidate entity:
{json.dumps(candidate_data, indent=2)}

Existing canonical entities:
{json.dumps(existing_data, indent=2)}

Important rules:

1. Compare entity identity, not merely textual similarity.

2. Entities must have compatible entity types.

3. Pay close attention to model versions.
   For example:
   - Stable Diffusion v1.5 and SD 1.5 may be the same entity.
   - Stable Diffusion v1.5 and Stable Diffusion v2.1 are different entities.
   - Stable Diffusion XL and Stable Diffusion v1.5 are different entities.

4. Abbreviations, aliases, punctuation, capitalization, and naming variants
   may refer to the same entity.

5. Do not merge entities merely because they belong to the same model family.

6. Use descriptions, aliases, properties, versions, organizations, and other
   available context when making the decision.

7. If there is strong evidence that the candidate matches exactly one existing
   entity, return "merge".

8. If the candidate clearly represents a distinct entity, return "new".

9. If the evidence is ambiguous, return "review".

10. canonical_entity_id MUST be one of the provided existing entity IDs when
    action is "merge".

11. canonical_entity_id MUST be null for "new".

12. Prefer "review" rather than guessing when identity is uncertain.

Return ONLY valid JSON:

{{
  "action": "merge" | "new" | "review",
  "canonical_entity_id": "existing_entity_id_or_null",
  "confidence": 0.0,
  "reason": "Short explanation",
  "matched_name": "matched canonical name or null"
}}
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.output_text.strip()

        data = json.loads(raw_output)

        decision = ResolutionDecision.model_validate(
            data
        )

        self._validate_decision(
            candidate=candidate,
            decision=decision,
            existing_entities=existing_entities,
        )

        return decision

    @staticmethod
    def _validate_decision(
        candidate: EntityCandidate,
        decision: ResolutionDecision,
        existing_entities: list[EntityCandidate],
    ) -> None:
        """
        Enforce hard safety constraints on LLM entity-resolution decisions.
        """

        # Build a lookup so we can inspect the entity selected by the LLM.
        entities_by_id = {
            entity.entity_id: entity
            for entity in existing_entities
        }

        # --------------------------------------------------
        # MERGE validation
        # --------------------------------------------------
        if decision.action == ResolutionAction.MERGE:

            # The LLM is only allowed to select an entity that
            # actually exists in the candidate set.
            if (
                decision.canonical_entity_id
                not in entities_by_id
            ):
                raise ValueError(
                    "LLM returned a canonical_entity_id "
                    "that does not exist."
                )

            matched_entity = entities_by_id[
                decision.canonical_entity_id
            ]

            # This is a HARD constraint.
            #
            # MODEL -> PROJECT merging is forbidden,
            # DATASET -> MODEL merging is forbidden, etc.
            if (
                matched_entity.entity_type
                != candidate.entity_type
            ):
                raise ValueError(
                    "Entity-resolution type mismatch: "
                    f"candidate type={candidate.entity_type.value}, "
                    f"matched type={matched_entity.entity_type.value}"
                )

        # --------------------------------------------------
        # NEW validation
        # --------------------------------------------------
        if decision.action == ResolutionAction.NEW:
            if decision.canonical_entity_id is not None:
                raise ValueError(
                    "NEW decisions must not contain "
                    "canonical_entity_id."
                )