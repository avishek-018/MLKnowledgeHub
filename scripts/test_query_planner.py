"""Smoke-test query planning workflows."""


"""Test LLM-based query planning."""

from dotenv import load_dotenv

from ml_knowledge_hub.agents.planner_agent import PlannerAgent


QUESTIONS = [
    # Graph
    "What models belong to NYUAD?",
    "What data was the AI-generated images detector evaluated on?",

    # Hybrid
    "Give me an overview of the NYUAD detector.",
    "What do we know about GenImage as a project?",

    # Unseen semantic artifact requests
    "What did the evaluation documents conclude about the pilot?",
    "Do we have any evidence that the CNN detector was reproduced?",
    "What implementation guidance is available for DIRE?",
    "What happened during the synthetic-image deployment incident?",
    "Describe the experimental findings for the pilot.",
    "What is the overall scope of the synthetic image screening project?",

    # Broad semantic — should NOT force a document type
    "Which projects are related to diffusion-generated images?",
]


def main():
    # --------------------------------------------------
    # Load OpenAI configuration from .env.
    # --------------------------------------------------
    load_dotenv()

    planner = PlannerAgent()

    for question in QUESTIONS:
        print()
        print("=" * 80)
        print("QUESTION:")
        print(question)

        plan = planner.plan(
            question
        )

        print("\nPLAN:")
        print(
            plan.model_dump(
                mode="json"
            )
        )


if __name__ == "__main__":
    main()