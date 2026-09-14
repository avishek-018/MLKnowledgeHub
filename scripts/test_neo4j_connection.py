"""Check connectivity to the configured Neo4j graph store."""


"""Verify Neo4j connectivity."""

from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)


def main():
    # --------------------------------------------------
    # Load credentials from local .env.
    # --------------------------------------------------
    load_dotenv()

    store = Neo4jStore()

    try:
        # --------------------------------------------------
        # Verify that:
        # - the database is reachable
        # - the URI is correct
        # - authentication succeeds
        # --------------------------------------------------
        store.verify_connection()

        print(
            "Neo4j connection successful."
        )

    finally:
        # Always close the driver.
        store.close()


if __name__ == "__main__":
    main()