"""Registry for canonical knowledge graph entities."""

"""Canonical entity registry for knowledge-graph resolution."""

import json
from pathlib import Path

from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
)


class EntityRegistry:
    def __init__(
        self,
        registry_path: str | Path = "data/knowledge_graph/entities.json",
    ):
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.registry_path.exists():
            self._write(
                {
                    "entities": [],
                    "resolution_history": [],
                }
            )

    def _read(self) -> dict:
        with self.registry_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with self.registry_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def list_entities(self) -> list[EntityCandidate]:
        data = self._read()

        return [
            EntityCandidate.model_validate(record)
            for record in data.get("entities", [])
        ]

    def get_entity(
        self,
        entity_id: str,
    ) -> EntityCandidate | None:
        for entity in self.list_entities():
            if entity.entity_id == entity_id:
                return entity

        return None

    def add_entity(
        self,
        entity: EntityCandidate,
    ) -> None:
        data = self._read()

        entities = data.setdefault(
            "entities",
            [],
        )

        if any(
            record["entity_id"] == entity.entity_id
            for record in entities
        ):
            raise ValueError(
                f"Entity already exists: {entity.entity_id}"
            )

        entities.append(
            entity.model_dump(mode="json")
        )

        self._write(data)

    def update_entity(
        self,
        entity: EntityCandidate,
    ) -> None:
        data = self._read()

        entities = data.setdefault(
            "entities",
            [],
        )

        for index, record in enumerate(entities):
            if record["entity_id"] == entity.entity_id:
                entities[index] = entity.model_dump(
                    mode="json"
                )

                self._write(data)
                return

        raise KeyError(
            f"Entity not found: {entity.entity_id}"
        )

    def add_alias(
        self,
        entity_id: str,
        alias: str,
    ) -> None:
        entity = self.get_entity(entity_id)

        if entity is None:
            raise KeyError(
                f"Entity not found: {entity_id}"
            )

        if alias not in entity.aliases:
            entity.aliases.append(alias)

        self.update_entity(entity)

    def record_resolution(
        self,
        record: dict,
    ) -> None:
        data = self._read()

        history = data.setdefault(
            "resolution_history",
            [],
        )

        history.append(record)

        self._write(data)

    def find_entity(
        self,
        text: str,
        entity_type=None,
    ):
        """
        Find a canonical entity mentioned in natural-language text.

        Matching priority:
        1. Exact normalized ID/name/alias
        2. Full ID/name/alias appearing inside the query
        3. Unique distinctive-token match

        The optional entity_type prevents, for example, a MODEL
        from being returned when we are looking for a PROJECT.
        """

        from ml_knowledge_hub.knowledge_graph.normalizer import (
            normalize_entity_name,
        )

        normalized_text = normalize_entity_name(text)

        # --------------------------------------------------
        # Candidate entities restricted by type when supplied.
        # --------------------------------------------------
        entities = [
            entity
            for entity in self.list_entities()
            if (
                entity_type is None
                or entity.entity_type == entity_type
            )
        ]

        partial_matches = []

        # --------------------------------------------------
        # 1. Exact/full-string matching.
        # --------------------------------------------------
        for entity in entities:

            candidate_strings = [
                entity.entity_id,
                entity.name,
                *entity.aliases,
            ]

            for candidate_string in candidate_strings:

                normalized_candidate = normalize_entity_name(
                    candidate_string
                )

                # Exact normalized match.
                if normalized_candidate == normalized_text:
                    return entity

                # Full entity name/alias occurs inside
                # a longer natural-language question.
                if normalized_candidate in normalized_text:
                    partial_matches.append(
                        (
                            len(normalized_candidate),
                            entity,
                        )
                    )

        # Prefer the longest/fullest match.
        if partial_matches:
            partial_matches.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return partial_matches[0][1]

        # --------------------------------------------------
        # 2. Unique distinctive-token fallback.
        #
        # Example:
        #
        #   query:
        #       "Which models are used by the NYUAD project?"
        #
        #   entity:
        #       "NYUAD AI-generated Images Detector"
        #
        # The unique token "nyuad" is enough to identify the
        # project even though the full name is not in the query.
        # --------------------------------------------------
        query_tokens = set(
            normalized_text.split()
        )

        token_matches = []

        # Very common words should not identify an entity.
        ignored_tokens = {
            "model",
            "models",
            "project",
            "projects",
            "dataset",
            "datasets",
            "image",
            "images",
            "detector",
            "detection",
            "generated",
            "used",
            "uses",
            "use",
            "which",
            "what",
            "the",
            "by",
            "for",
            "are",
            "is",
        }

        for entity in entities:

            # Build searchable text from canonical ID,
            # display name, and known aliases.
            candidate_strings = [
                entity.entity_id,
                entity.name,
                *entity.aliases,
            ]

            entity_tokens = set()

            for candidate_string in candidate_strings:
                entity_tokens.update(
                    normalize_entity_name(
                        candidate_string
                    ).split()
                )

            # Only use reasonably distinctive tokens.
            distinctive_tokens = {
                token
                for token in entity_tokens
                if (
                    len(token) >= 4
                    and token not in ignored_tokens
                )
            }

            overlap = (
                query_tokens
                & distinctive_tokens
            )

            if overlap:
                token_matches.append(
                    (
                        len(overlap),
                        entity,
                    )
                )

        # --------------------------------------------------
        # Only accept the fallback if exactly one entity
        # has the best score.
        #
        # Ambiguous matches return None instead of guessing.
        # --------------------------------------------------
        if token_matches:

            best_score = max(
                score
                for score, _ in token_matches
            )

            best_matches = [
                entity
                for score, entity in token_matches
                if score == best_score
            ]

            if len(best_matches) == 1:
                return best_matches[0]

        return None