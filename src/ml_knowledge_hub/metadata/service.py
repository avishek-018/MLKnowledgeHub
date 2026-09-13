"""Metadata service utilities."""


class MetadataService:
    def __init__(self, registry):
        self.registry = registry

    def list_projects(self) -> list[str]:
        return self.registry.list_projects()

    def count_projects(self) -> int:
        return self.registry.count_projects()

    def list_asset_types(self) -> list[str]:
        return sorted({
            record["asset_type"]
            for record in self.registry.list_assets()
            if record.get("asset_type")
        })

    def count_assets_by_type(self) -> dict[str, int]:
        counts = {}

        for record in self.registry.list_assets():
            asset_type = record.get("asset_type")

            if not asset_type:
                continue

            counts[asset_type] = (
                counts.get(asset_type, 0) + 1
            )

        return counts

    def get_assets_for_project(
        self,
        project_id: str,
    ):
        return self.registry.get_assets_for_project(
            project_id
        )

    def get_assets_by_type(
        self,
        asset_type: str,
    ):
        return self.registry.get_assets_by_type(
            asset_type
        )