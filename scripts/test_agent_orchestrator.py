"""Test the complete Planner -> Evaluator -> Revision workflow."""

from dotenv import load_dotenv

from ml_knowledge_hub.agents.orchestrator import (
    AgentOrchestrator,
)


QUESTIONS = [
    # Semantic search constrained to model-card documents.
    "Which model cards describe AI image detectors?",

    # Direct graph relationship.
    "What models belong to NYUAD?",

    # Another direct graph relationship.
    "What data was the AI-generated images detector evaluated on?",

    # Hybrid entity-context question.
    "Give me an overview of the NYUAD detector.",

    # Semantic document question.
    "Do we have any evidence that the CNN detector was reproduced?",

    # Broad semantic discovery.
    "Which projects are related to diffusion-generated images?",
]


def print_plan(
    label: str,
    plan,
):
    print(f"\n{label}:")
    print(
        plan.model_dump(
            mode="json"
        )
    )


def main():
    load_dotenv()

    orchestrator = AgentOrchestrator()

    for question in QUESTIONS:
        print()
        print("=" * 100)

        print("QUESTION:")
        print(question)

        result = orchestrator.create_plan(
            query=question
        )

        print_plan(
            "INITIAL PLAN",
            result["initial_plan"],
        )

        print("\nEVALUATION:")
        print(
            result["evaluation"].model_dump(
                mode="json"
            )
        )

        print("\nREVISED:")
        print(
            result["revised"]
        )

        print_plan(
            "FINAL PLAN",
            result["final_plan"],
        )


if __name__ == "__main__":
    main()
