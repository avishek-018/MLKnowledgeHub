"""Inspect entities that remain unresolved in the knowledge graph."""


"""Inspect unresolved entity-resolution decisions."""

import json
from pathlib import Path


ENTITY_REGISTRY_PATH = Path(
    "data/knowledge_graph/entities.json"
)


def main():
    # --------------------------------------------------
    # Load the canonical entity registry.
    #
    # This file contains:
    # - canonical entities
    # - resolution history
    # --------------------------------------------------
    with ENTITY_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    history = data.get(
        "resolution_history",
        [],
    )

    unresolved = []

    # --------------------------------------------------
    # Keep only decisions that ended in REVIEW.
    # --------------------------------------------------
    for record in history:
        decision = record.get(
            "decision",
            {}
        )

        if decision.get("action") == "review":
            unresolved.append(
                record
            )

    print(
        f"Unresolved resolution records: "
        f"{len(unresolved)}"
    )

    print()

    # --------------------------------------------------
    # Print each unresolved candidate together with the
    # resolver's confidence and explanation.
    # --------------------------------------------------
    for index, record in enumerate(
        unresolved,
        start=1,
    ):
        candidate = record.get(
            "candidate",
            {}
        )

        decision = record.get(
            "decision",
            {}
        )

        print("=" * 70)

        print(
            f"[{index}/{len(unresolved)}]"
        )

        print(
            "Candidate ID:",
            candidate.get("entity_id"),
        )

        print(
            "Name:",
            candidate.get("name"),
        )

        print(
            "Type:",
            candidate.get("entity_type"),
        )

        print(
            "Description:",
            candidate.get("description"),
        )

        print(
            "Confidence:",
            decision.get("confidence"),
        )

        print(
            "Matched name:",
            decision.get("matched_name"),
        )

        print(
            "Reason:",
            decision.get("reason"),
        )

        print()


if __name__ == "__main__":
    main()

