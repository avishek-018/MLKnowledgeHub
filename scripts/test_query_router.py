"""Test the agentic QueryRouter."""

from dotenv import load_dotenv

from ml_knowledge_hub.query.router import QueryRouter


QUESTIONS = [
    "Which model cards describe AI image detectors?",
    "What models belong to NYUAD?",
    "What data was the AI-generated images detector evaluated on?",
    "Give me an overview of the NYUAD detector.",
    "Do we have any evidence that the CNN detector was reproduced?",
    "Which projects are related to diffusion-generated images?",
    "How many projects do we have?",
]


def main():
    load_dotenv()

    router = QueryRouter()

    for question in QUESTIONS:
        print()
        print("=" * 100)

        print("QUESTION:")
        print(question)

        plan = router.route(
            question
        )

        print("\nFINAL EXECUTION PLAN:")
        print(plan)


if __name__ == "__main__":
    main()
