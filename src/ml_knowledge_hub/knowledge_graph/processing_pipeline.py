"""Pipeline orchestration for knowledge graph processing."""

"""Process extracted KG data through entity resolution."""

from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    KGEntity,
    KGExtractionResult,
    KGRelation,
)
from ml_knowledge_hub.knowledge_graph.processing_schema import (
    ResolvedKGResult,
)
from ml_knowledge_hub.knowledge_graph.resolution_pipeline import (
    EntityResolutionPipeline,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
    ResolutionAction,
)

from ml_knowledge_hub.knowledge_graph.canonical_ids import (
    canonical_id_for_new_entity,
)

class KGProcessingPipeline:
    """
    Convert raw LLM-extracted KG data into canonical KG data.

    Pipeline:

        raw entities
            ↓
        entity resolution
            ↓
        canonical entity IDs
            ↓
        relation ID rewriting
            ↓
        resolved KG result
    """

    def __init__(
        self,
        resolution_pipeline: EntityResolutionPipeline,
        registry: EntityRegistry,
    ):
        self.resolution_pipeline = resolution_pipeline
        self.registry = registry

    def process(
        self,
        extraction: KGExtractionResult,
    ) -> ResolvedKGResult:
        """
        Resolve every extracted entity and rewrite relations so they
        use canonical entity IDs.
        """

        # Maps the temporary/raw entity IDs produced by the LLM
        # to canonical entity IDs used by our knowledge graph.
        entity_id_map: dict[str, str] = {}

        # Keep track of entities that require manual/future review.
        unresolved_entity_ids: list[str] = []

        # --------------------------------------------------
        # 1. Resolve every extracted entity
        # --------------------------------------------------
        for entity in extraction.entities:

            # Convert the extractor representation into the
            # representation expected by the entity resolver.
            candidate = EntityCandidate(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties=entity.properties,
            )

            decision = self.resolution_pipeline.resolve_entity(
                candidate
            )

            # --------------------------------------------------
            # Existing canonical entity
            # --------------------------------------------------
            if decision.action == ResolutionAction.MERGE:

                canonical_id = (
                    decision.canonical_entity_id
                )

                if canonical_id is None:
                    raise ValueError(
                        "MERGE decision has no "
                        "canonical_entity_id."
                    )

                entity_id_map[
                    entity.entity_id
                ] = canonical_id

            # --------------------------------------------------
            # Newly created canonical entity
            # --------------------------------------------------
            elif decision.action == ResolutionAction.NEW:

                # Reconstruct the exact canonical ID chosen by
                # EntityResolutionPipeline.
                #
                # Because the entity has already been inserted,
                # this helper will deterministically return the
                # same ID.
                canonical_id = canonical_id_for_new_entity(
                    candidate=candidate,
                    registry=self.registry,
                )

                canonical_entity = self.registry.get_entity(
                    canonical_id
                )

                if canonical_entity is None:
                    raise RuntimeError(
                        "NEW entity was not found in the "
                        "canonical registry. "
                        f"Candidate: {candidate.name!r}, "
                        f"type={candidate.entity_type.value}, "
                        f"expected_id={canonical_id!r}"
                    )

                entity_id_map[
                    entity.entity_id
                ] = canonical_entity.entity_id
            # --------------------------------------------------
            # Unresolved entity
            # --------------------------------------------------
            elif decision.action == ResolutionAction.REVIEW:

                # Do not create a canonical graph node yet.
                unresolved_entity_ids.append(
                    entity.entity_id
                )

        # --------------------------------------------------
        # 2. Build canonical entity objects
        # --------------------------------------------------
        resolved_entities: list[KGEntity] = []

        # Multiple raw mentions can resolve to the same canonical
        # entity, so prevent duplicate nodes in this result.
        seen_entity_ids: set[str] = set()

        for canonical_id in entity_id_map.values():

            if canonical_id in seen_entity_ids:
                continue

            canonical = self.registry.get_entity(
                canonical_id
            )

            if canonical is None:
                continue

            resolved_entities.append(
                KGEntity(
                    entity_id=canonical.entity_id,
                    name=canonical.name,
                    entity_type=canonical.entity_type,
                    description=canonical.description,
                    properties=canonical.properties,
                )
            )

            seen_entity_ids.add(
                canonical_id
            )

        # --------------------------------------------------
        # 3. Rewrite relations using canonical IDs
        # --------------------------------------------------
        resolved_relations: list[KGRelation] = []

        for relation in extraction.relations:

            source_id = entity_id_map.get(
                relation.source_id
            )

            target_id = entity_id_map.get(
                relation.target_id
            )

            # If either endpoint is unresolved, we cannot safely
            # create this relation in the canonical graph.
            if source_id is None or target_id is None:
                continue

            resolved_relations.append(
                KGRelation(
                    source_id=source_id,
                    relation_type=relation.relation_type,
                    target_id=target_id,
                    evidence=relation.evidence,
                )
            )

        return ResolvedKGResult(
            entities=resolved_entities,
            relations=resolved_relations,
            entity_id_map=entity_id_map,
            unresolved_entity_ids=unresolved_entity_ids,
        )

    def _find_new_entity(
        self,
        candidate: EntityCandidate,
    ) -> EntityCandidate | None:
        """
        Find the canonical entity created for a NEW decision.

        We compare the stored canonical name with the candidate name.
        """

        for entity in self.registry.list_entities():

            if (
                entity.name == candidate.name
                and entity.entity_type
                == candidate.entity_type
            ):
                return entity

        return None