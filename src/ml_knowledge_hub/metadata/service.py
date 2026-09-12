"""Metadata service utilities."""

class MetadataService:
    def __init__(self, documents):
        self.documents = documents

    def list_projects(self) -> list[str]:
        return sorted({
            doc.project_id
            for doc in self.documents
        })

    def count_projects(self) -> int:
        return len(self.list_projects())

    def list_asset_types(self) -> list[str]:
        return sorted({
            doc.asset_type
            for doc in self.documents
        })

    def count_assets_by_type(self) -> dict[str, int]:
        counts = {}

        for doc in self.documents:
            counts[doc.asset_type] = (
                counts.get(doc.asset_type, 0) + 1
            )

        return counts

    def get_assets_for_project(
        self,
        project_id: str,
    ):
        return [
            doc
            for doc in self.documents
            if doc.project_id == project_id
        ]

    def get_assets_by_type(
        self,
        asset_type: str,
    ):
        return [
            doc
            for doc in self.documents
            if doc.asset_type == asset_type
        ]