"""Evaluate the Planner -> Evaluator -> Revision agent workflow."""

import json
from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.agents.evaluator_agent import EvaluatorAgent
from ml_knowledge_hub.agents.planner_agent import PlannerAgent
from ml_knowledge_hub.agents.revision_agent import RevisionAgent


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BENCHMARK_PATH = Path(
    "data/evaluation/query_planning_benchmark.json"
)

OUTPUT_PATH = Path(
    "data/evaluation/query_planning_results.json"
)


def normalize_asset_types(asset_types):
    """
    Convert asset-type enums/strings into a comparable sorted list.

    Sorting makes:
        ["paper", "model_card"]

    equivalent to:
        ["model_card", "paper"]
    """

    if not asset_types:
        return None

    normalized = []

    for asset_type in asset_types:
        if hasattr(asset_type, "value"):
            normalized.append(asset_type.value)
        else:
            normalized.append(asset_type)

    return sorted(normalized)


def enum_value(value):
    """
    Convert enum values to normal strings for comparison.
    """

    if value is None:
        return None

    if hasattr(value, "value"):
        return value.value

    return value


def plan_matches_reference(
    actual_query_type,
    actual_operation,
    actual_asset_types,
    reference,
):
    """
    Compare execution-critical plan fields against one
    acceptable reference plan.
    """

    expected_asset_types = normalize_asset_types(
        reference.get("asset_types")
    )

    return (
        actual_query_type
        == reference["query_type"]
        and
        actual_operation
        == reference["operation"]
        and
        actual_asset_types
        == expected_asset_types
    )

def evaluate_plan(
    plan,
    expected,
):
    """
    Compare a generated plan against benchmark expectations.

    Supports two benchmark formats:

    1. Single-reference:
       expected_query_type
       expected_operation
       expected_asset_types
       ...

    2. Multi-reference:
       acceptable_plans = [
           {...},
           {...},
       ]

    Metrics:
    - routing_correct:
        query_type + operation

    - execution_plan_correct:
        query_type + operation + asset_types

    - full_plan_correct:
        also checks entity annotations for single-reference cases.
        For ambiguous multi-reference cases, execution correctness
        is treated as sufficient.
    """

    actual_query_type = enum_value(
        plan.query_type
    )

    actual_operation = enum_value(
        plan.operation
    )

    actual_entity_type = enum_value(
        plan.entity_type
    )

    actual_target_entity_type = enum_value(
        plan.target_entity_type
    )

    actual_asset_types = normalize_asset_types(
        plan.asset_types
    )

    # --------------------------------------------------------------
    # Multi-reference evaluation
    # --------------------------------------------------------------
    acceptable_plans = expected.get(
        "acceptable_plans"
    )

    if acceptable_plans:

        # Routing is correct when query type + operation
        # match at least one acceptable interpretation.
        routing_correct = any(
            actual_query_type
            == reference["query_type"]
            and
            actual_operation
            == reference["operation"]
            for reference in acceptable_plans
        )

        # Execution correctness also requires asset filters
        # to match the same acceptable reference.
        execution_plan_correct = any(
            plan_matches_reference(
                actual_query_type=actual_query_type,
                actual_operation=actual_operation,
                actual_asset_types=actual_asset_types,
                reference=reference,
            )
            for reference in acceptable_plans
        )

        # Entity annotations are intentionally not scored
        # strictly for genuinely ambiguous benchmark items.
        full_plan_correct = execution_plan_correct

    # --------------------------------------------------------------
    # Single-reference evaluation
    # --------------------------------------------------------------
    else:
        expected_asset_types = normalize_asset_types(
            expected.get(
                "expected_asset_types"
            )
        )

        routing_correct = (
            actual_query_type
            == expected["expected_query_type"]
            and
            actual_operation
            == expected["expected_operation"]
        )

        execution_plan_correct = (
            routing_correct
            and
            actual_asset_types
            == expected_asset_types
        )

        full_plan_correct = (
            execution_plan_correct
            and
            actual_entity_type
            == expected.get(
                "expected_entity_type"
            )
            and
            actual_target_entity_type
            == expected.get(
                "expected_target_entity_type"
            )
        )

    return {
        "routing_correct": routing_correct,
        "execution_plan_correct": execution_plan_correct,
        "full_plan_correct": full_plan_correct,

        "actual": {
            "query_type": actual_query_type,
            "operation": actual_operation,
            "entity_type": actual_entity_type,
            "target_entity_type": (
                actual_target_entity_type
            ),
            "asset_types": actual_asset_types,
        },
    }


def percentage(
    numerator,
    denominator,
):
    """
    Safely calculate a percentage.
    """

    if denominator == 0:
        return 0.0

    return round(
        (numerator / denominator) * 100,
        2,
    )




def main():
    load_dotenv()

    # --------------------------------------------------------------
    # Load benchmark.
    # --------------------------------------------------------------
    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = json.load(file)

    planner = PlannerAgent()
    evaluator = EvaluatorAgent()
    revision_agent = RevisionAgent()

    results = []

    # --------------------------------------------------------------
    # Counters
    # --------------------------------------------------------------
    initial_routing_correct = 0
    initial_execution_correct = 0
    initial_full_correct = 0

    final_routing_correct = 0
    final_execution_correct = 0
    final_full_correct = 0

    planner_errors = 0
    critic_detected_errors = 0

    correct_plans_rejected = 0

    revisions_attempted = 0
    revisions_successful = 0

    print(
        f"Evaluating {len(benchmark)} benchmark queries..."
    )

    for index, item in enumerate(
        benchmark,
        start=1,
    ):
        query = item["query"]

        print()
        print("=" * 100)
        print(
            f"[{index}/{len(benchmark)}] "
            f"{item['id']}: {query}"
        )

        # ----------------------------------------------------------
        # 1. Initial Planner Agent
        # ----------------------------------------------------------
        initial_plan = planner.plan(
            query=query
        )

        initial_eval = evaluate_plan(
            plan=initial_plan,
            expected=item,
        )

        if initial_eval["routing_correct"]:
            initial_routing_correct += 1
        if initial_eval["execution_plan_correct"]:
            initial_execution_correct += 1
        if initial_eval["full_plan_correct"]:
            initial_full_correct += 1
        else:
            planner_errors += 1

        # ----------------------------------------------------------
        # 2. Evaluator Agent
        # ----------------------------------------------------------
        evaluation = evaluator.evaluate(
            query=query,
            plan=initial_plan,
        )

        # ----------------------------------------------------------
        # Did the critic detect an actually incorrect plan?
        # ----------------------------------------------------------
        if (
            not initial_eval["full_plan_correct"]
            and not evaluation.valid
        ):
            critic_detected_errors += 1

        # ----------------------------------------------------------
        # Did the critic reject a benchmark-correct plan?
        #
        # This is useful because an evaluator can hurt performance
        # if it unnecessarily criticizes correct plans.
        # ----------------------------------------------------------
        if (
            initial_eval["full_plan_correct"]
            and not evaluation.valid
        ):
            correct_plans_rejected += 1

        # ----------------------------------------------------------
        # 3. Revision Agent
        # ----------------------------------------------------------
        if evaluation.valid:
            final_plan = initial_plan
            revised = False

        else:
            revisions_attempted += 1

            final_plan = revision_agent.revise(
                query=query,
                original_plan=initial_plan,
                evaluation=evaluation,
            )

            revised = True

        # ----------------------------------------------------------
        # 4. Evaluate final plan.
        # ----------------------------------------------------------
        final_eval = evaluate_plan(
            plan=final_plan,
            expected=item,
        )

        if final_eval["routing_correct"]:
            final_routing_correct += 1
        if final_eval["execution_plan_correct"]:
            final_execution_correct += 1
        if final_eval["full_plan_correct"]:
            final_full_correct += 1

        # ----------------------------------------------------------
        # Successful revision means:
        #
        # initial plan was wrong
        # AND final plan became correct.
        # ----------------------------------------------------------
        revision_success = (
            revised
            and not initial_eval[
                "full_plan_correct"
            ]
            and final_eval[
                "full_plan_correct"
            ]
        )

        if revision_success:
            revisions_successful += 1

        # ----------------------------------------------------------
        # Save complete per-query information.
        # ----------------------------------------------------------
        results.append(
            {
                "id": item["id"],
                "query": query,

                "expected": (
                    {
                        "acceptable_plans": item[
                            "acceptable_plans"
                        ]
                    }
                    if item.get("acceptable_plans")
                    else
                    {
                        "query_type": item[
                            "expected_query_type"
                        ],
                        "operation": item[
                            "expected_operation"
                        ],
                        "entity_type": item.get(
                            "expected_entity_type"
                        ),
                        "target_entity_type": item.get(
                            "expected_target_entity_type"
                        ),
                        "asset_types": normalize_asset_types(
                            item.get(
                                "expected_asset_types"
                            )
                        ),
                    }
                ),

                "initial_plan": (
                    initial_plan.model_dump(
                        mode="json"
                    )
                ),

                "initial_routing_correct": (
                    initial_eval[
                        "routing_correct"
                    ]
                ),
                "initial_execution_plan_correct": (
                    initial_eval[
                        "execution_plan_correct"
                    ]
                ),
                "initial_full_plan_correct": (
                    initial_eval[
                        "full_plan_correct"
                    ]
                ),

                "evaluation": (
                    evaluation.model_dump(
                        mode="json"
                    )
                ),

                "revised": revised,

                "final_plan": (
                    final_plan.model_dump(
                        mode="json"
                    )
                ),

                "final_routing_correct": (
                    final_eval[
                        "routing_correct"
                    ]
                ),
                "final_execution_plan_correct": (
                    final_eval[
                        "execution_plan_correct"
                    ]
                ),
                "final_full_plan_correct": (
                    final_eval[
                        "full_plan_correct"
                    ]
                ),

                "revision_success": (
                    revision_success
                ),
            }
        )

        print(
            "Initial route correct:",
            initial_eval["routing_correct"],
        )

        print(
            "Initial full plan correct:",
            initial_eval["full_plan_correct"],
        )

        print(
            "Evaluator valid:",
            evaluation.valid,
        )

        print(
            "Revised:",
            revised,
        )

        print(
            "Final route correct:",
            final_eval["routing_correct"],
        )

        print(
            "Final full plan correct:",
            final_eval["full_plan_correct"],
        )

    total = len(
        benchmark
    )

    # --------------------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------------------
    summary = {
        "total_queries": total,

        "initial_routing_accuracy": percentage(
            initial_routing_correct,
            total,
        ),

        "initial_execution_plan_accuracy": percentage(
            initial_execution_correct,
            total,
        ),

        "initial_full_plan_accuracy": percentage(
            initial_full_correct,
            total,
        ),

        "final_routing_accuracy": percentage(
            final_routing_correct,
            total,
        ),
        "final_execution_plan_accuracy": percentage(
            final_execution_correct,
            total,
        ),
        "final_full_plan_accuracy": percentage(
            final_full_correct,
            total,
        ),

        "planner_errors": planner_errors,

        "critic_detection_rate": percentage(
            critic_detected_errors,
            planner_errors,
        ),

        "correct_plans_rejected_by_critic": (
            correct_plans_rejected
        ),

        "revisions_attempted": (
            revisions_attempted
        ),

        "successful_revisions": (
            revisions_successful
        ),

        "revision_success_rate": percentage(
            revisions_successful,
            revisions_attempted,
        ),
    }

    # --------------------------------------------------------------
    # Save detailed results.
    # --------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "summary": summary,
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    # --------------------------------------------------------------
    # Print final summary.
    # --------------------------------------------------------------
    print()
    print("=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    print(
        f"Queries: {total}"
    )

    print(
        "Initial routing accuracy: "
        f"{summary['initial_routing_accuracy']}%"
    )

    print(
        "Initial full-plan accuracy: "
        f"{summary['initial_full_plan_accuracy']}%"
    )
    print(
        "Initial execution-plan accuracy: "
        f"{summary['initial_execution_plan_accuracy']}%"
    )
    print(
        "Final routing accuracy: "
        f"{summary['final_routing_accuracy']}%"
    )
    print(
        "Final execution-plan accuracy: "
        f"{summary['final_execution_plan_accuracy']}%"
    )
    print(
        "Final full-plan accuracy: "
        f"{summary['final_full_plan_accuracy']}%"
    )

    print(
        "Critic detection rate: "
        f"{summary['critic_detection_rate']}%"
    )

    print(
        "Correct plans rejected by critic: "
        f"{summary['correct_plans_rejected_by_critic']}"
    )

    print(
        "Revision success rate: "
        f"{summary['revision_success_rate']}%"
    )

    print()
    print(
        f"Detailed results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()