"""Smoke-test entry point for entity resolution workflows."""


from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.entity_resolver import (
    EntityResolver,
)
from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
)
from ml_knowledge_hub.knowledge_graph.resolution_schema import (
    EntityCandidate,
)


def main():
    load_dotenv()

    existing_entities = [
        EntityCandidate(
            entity_id="stable_diffusion_v1_5",
            name="Stable Diffusion v1.5",
            entity_type=EntityType.MODEL,
            aliases=[
                "Stable Diffusion 1.5",
            ],
            description=(
                "Stable Diffusion version 1.5 "
                "text-to-image diffusion model."
            ),
        ),
        EntityCandidate(
            entity_id="stable_diffusion_v2_1",
            name="Stable Diffusion v2.1",
            entity_type=EntityType.MODEL,
            description=(
                "Stable Diffusion version 2.1 "
                "text-to-image diffusion model."
            ),
        ),
        EntityCandidate(
            entity_id="stable_diffusion_xl",
            name="Stable Diffusion XL",
            entity_type=EntityType.MODEL,
            aliases=[
                "SDXL",
            ],
            description=(
                "Stable Diffusion XL image generation model."
            ),
        ),
    ]

    candidate = EntityCandidate(
        entity_id="sd_new",
        name="SD New",
        entity_type=EntityType.MODEL,
        description=(
            ""
        ),
    )

    resolver = EntityResolver()

    decision = resolver.resolve(
        candidate=candidate,
        existing_entities=existing_entities,
    )

    print("\nENTITY RESOLUTION")
    print("-----------------")
    print(f"Candidate: {candidate.name}")
    print(f"Action: {decision.action.value}")
    print(
        "Canonical ID:",
        decision.canonical_entity_id,
    )
    print(
        "Confidence:",
        decision.confidence,
    )
    print(
        "Matched name:",
        decision.matched_name,
    )
    print(
        "Reason:",
        decision.reason,
    )


if __name__ == "__main__":
    main()