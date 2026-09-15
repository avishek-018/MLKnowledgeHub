"""Neo4j graph store integration."""

"""Neo4j connection wrapper."""

import os
import json
from neo4j import GraphDatabase


class Neo4jStore:
    """
    Small wrapper around the Neo4j Python driver.

    For now this class only handles:
    - connection setup
    - connection verification
    - clean shutdown

    Graph writes will be added in the next step.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        # --------------------------------------------------
        # Read credentials from arguments first.
        #
        # If they are not supplied explicitly, fall back to
        # environment variables loaded from .env.
        # --------------------------------------------------
        self.uri = uri or os.getenv(
            "NEO4J_URI"
        )

        self.username = username or os.getenv(
            "NEO4J_USERNAME"
        )

        self.password = password or os.getenv(
            "NEO4J_PASSWORD"
        )

        # --------------------------------------------------
        # Validate configuration before trying to connect.
        # --------------------------------------------------
        missing = []

        if not self.uri:
            missing.append("NEO4J_URI")

        if not self.username:
            missing.append("NEO4J_USERNAME")

        if not self.password:
            missing.append("NEO4J_PASSWORD")

        if missing:
            raise ValueError(
                "Missing Neo4j configuration: "
                + ", ".join(missing)
            )

        # --------------------------------------------------
        # Create the Neo4j driver.
        # --------------------------------------------------
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(
                self.username,
                self.password,
            ),
        )

    def verify_connection(self) -> None:
        """
        Ask the Neo4j driver to verify that the database
        is reachable and the credentials are valid.
        """

        self.driver.verify_connectivity()

    def close(self) -> None:
        """
        Close the Neo4j driver cleanly.
        """

        self.driver.close()

    def upsert_entity(self, entity) -> None:
        """
        Insert or update one canonical KG entity in Neo4j.

        The entity_id is treated as the stable unique identifier.
        If a node with the same entity_id already exists, its
        properties are updated instead of creating a duplicate.
        """

        # Use the entity type as the Neo4j node label.
        #
        # Example:
        #   EntityType.MODEL -> :Model
        #   EntityType.DATASET -> :Dataset
        label = entity.entity_type.value.capitalize()

        # Neo4j labels cannot be safely passed as query parameters,
        # so we construct the label from our controlled Enum value.
        query = f"""
        MERGE (e:{label} {{entity_id: $entity_id}})
        SET
            e.name = $name,
            e.description = $description,
            e.entity_type = $entity_type,
            e.properties_json = $properties_json
        """

        

        # Run the write transaction.
        with self.driver.session() as session:
            session.run(
                query,
                entity_id=entity.entity_id,
                name=entity.name,
                description=entity.description,
                entity_type=entity.entity_type.value,
                properties_json=json.dumps(
                    entity.properties
                ),
            )

    def upsert_relation(self, relation) -> None:
        """
        Insert or update one canonical KG relation in Neo4j.

        The relation assumes both endpoint nodes already exist.
        """

        # --------------------------------------------------
        # Relationship type comes from our controlled Enum.
        #
        # Examples:
        #   USES_MODEL
        #   EVALUATED_ON
        #   REPORTS
        # --------------------------------------------------
        relation_type = relation.relation_type.value

        query = f"""
        MATCH (source {{entity_id: $source_id}})
        MATCH (target {{entity_id: $target_id}})

        MERGE (source)-[r:{relation_type}]->(target)

        SET
            r.evidence = $evidence
        """

        # --------------------------------------------------
        # Execute the relation upsert.
        #
        # MERGE prevents duplicate relationships of the same
        # type between the same source and target nodes.
        # --------------------------------------------------
        with self.driver.session() as session:
            session.run(
                query,
                source_id=relation.source_id,
                target_id=relation.target_id,
                evidence=relation.evidence,
            )

    def write_result(self, result) -> None:
        """
        Write a complete resolved KG result to Neo4j.

        The method:
        1. upserts all canonical entities
        2. upserts all canonical relations

        This keeps Neo4j synchronized with the already-resolved
        knowledge graph produced by the processing pipeline.
        """

        # --------------------------------------------------
        # 1. Write all canonical entities first.
        #
        # Relations depend on these nodes already existing.
        # --------------------------------------------------
        for entity in result.entities:
            self.upsert_entity(
                entity
            )

        # --------------------------------------------------
        # 2. Write all canonical relations.
        #
        # Because upsert_relation() uses MERGE, rerunning this
        # method should not create duplicate relationships.
        # --------------------------------------------------
        for relation in result.relations:
            self.upsert_relation(
                relation
            )

    def get_models_for_project(
        self,
        project_id: str,
    ) -> list[dict]:
        """
        Return models connected to a project through USES_MODEL.

        Example:
            Project --USES_MODEL--> Model
        """

        query = """
        MATCH (p:Project {entity_id: $project_id})
            -[:USES_MODEL]->
            (m:Model)

        RETURN
            m.entity_id AS entity_id,
            m.name AS name,
            m.description AS description

        ORDER BY m.name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                project_id=project_id,
            )

            return [
                record.data()
                for record in result
            ]


    def get_projects_for_model(
        self,
        model_id: str,
    ) -> list[dict]:
        """
        Return projects that use a particular model.

        Example:
            Project --USES_MODEL--> Model
        """

        query = """
        MATCH (p:Project)
            -[:USES_MODEL]->
            (m:Model {entity_id: $model_id})

        RETURN
            p.entity_id AS entity_id,
            p.name AS name,
            p.description AS description

        ORDER BY p.name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                model_id=model_id,
            )

            return [
                record.data()
                for record in result
            ]


    def get_datasets_for_model(
        self,
        model_id: str,
    ) -> list[dict]:
        """
        Return datasets associated with a model.

        We include both:
        - TRAINED_ON
        - EVALUATED_ON

        The relationship type is returned so callers can
        distinguish training data from evaluation data.
        """

        query = """
        MATCH (m:Model {entity_id: $model_id})
            -[r:TRAINED_ON|EVALUATED_ON]->
            (d:Dataset)

        RETURN
            d.entity_id AS entity_id,
            d.name AS name,
            d.description AS description,
            type(r) AS relationship

        ORDER BY d.name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                model_id=model_id,
            )

            return [
                record.data()
                for record in result
            ]


    def get_metrics_for_model(
        self,
        model_id: str,
    ) -> list[dict]:
        """
        Return metrics reported by a model.

        Example:
            Model --REPORTS--> Metric
        """

        query = """
        MATCH (m:Model {entity_id: $model_id})
            -[:REPORTS]->
            (metric:Metric)

        RETURN
            metric.entity_id AS entity_id,
            metric.name AS name,
            metric.description AS description,
            metric.properties_json AS properties_json

        ORDER BY metric.name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                model_id=model_id,
            )

            return [
                record.data()
                for record in result
            ]


    def get_models_for_dataset(
        self,
        dataset_id: str,
    ) -> list[dict]:
        """
        Return models connected to a dataset through
        TRAINED_ON or EVALUATED_ON.
        """

        query = """
        MATCH (m:Model)
            -[r:TRAINED_ON|EVALUATED_ON]->
            (d:Dataset {entity_id: $dataset_id})

        RETURN
            m.entity_id AS entity_id,
            m.name AS name,
            m.description AS description,
            type(r) AS relationship

        ORDER BY m.name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                dataset_id=dataset_id,
            )

            return [
                record.data()
                for record in result
            ]