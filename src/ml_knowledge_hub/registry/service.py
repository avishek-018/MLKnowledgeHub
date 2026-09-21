
"""JSON-backed asset registry."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AssetRegistry:
    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():
            self._write_registry(
                {
                    "records": [],
                    "updated_at": self._now(),
                }
            )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_registry(self) -> dict[str, Any]:
        with self.registry_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _write_registry(
        self,
        data: dict[str, Any],
    ) -> None:
        data["updated_at"] = self._now()

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

    def list_assets(self) -> list[dict[str, Any]]:
        data = self._read_registry()

        return data.get("records", [])

    def add_asset(
        self,
        record: dict[str, Any],
    ) -> None:
        data = self._read_registry()

        records = data.setdefault(
            "records",
            [],
        )

        records.append(record)

        self._write_registry(data)

    def get_assets_for_project(
        self,
        project_id: str,
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_assets()
            if record.get("project_id") == project_id
        ]

    def get_assets_by_type(
        self,
        asset_type: str,
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_assets()
            if record.get("asset_type") == asset_type
        ]

    def list_projects(self) -> list[str]:
        return sorted(
            {
                record["project_id"]
                for record in self.list_assets()
                if record.get("project_id")
            }
        )

    def count_projects(self) -> int:
        return len(self.list_projects())