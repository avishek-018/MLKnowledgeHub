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