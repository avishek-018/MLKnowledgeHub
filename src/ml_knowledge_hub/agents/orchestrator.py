"""Orchestrator for the agentic query-planning workflow."""

from ml_knowledge_hub.agents.evaluator_agent import EvaluatorAgent
from ml_knowledge_hub.agents.planner_agent import PlannerAgent
from ml_knowledge_hub.agents.revision_agent import RevisionAgent


class AgentOrchestrator:
    """
    Coordinate the Planner, Evaluator, and Revision agents.

    Workflow:

        user query
            ↓
        Planner Agent
            ↓
        initial plan
            ↓
        Evaluator Agent
            ↓
        valid?
        ├── yes → final plan = initial plan
        └── no  → Revision Agent → final plan

    Only one revision round is allowed for now.
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        evaluator: EvaluatorAgent | None = None,
        revision_agent: RevisionAgent | None = None,
    ):
        # --------------------------------------------------
        # Dependency injection keeps this class easy to test.
        #
        # Real agents are used by default, but fake agents can
        # later be passed during unit tests.
        # --------------------------------------------------
        self.planner = planner or PlannerAgent()
        self.evaluator = evaluator or EvaluatorAgent()
        self.revision_agent = revision_agent or RevisionAgent()

    def create_plan(
        self,
        query: str,
    ) -> dict:
        """
        Produce the final executable query plan.

        Returns all intermediate reasoning artifacts so the
        workflow can be inspected and evaluated later.
        """

        # --------------------------------------------------
        # Step 1: Planner Agent creates an initial plan.
        # --------------------------------------------------
        initial_plan = self.planner.plan(
            query=query
        )

        # --------------------------------------------------
        # Step 2: Evaluator Agent critiques that plan.
        # --------------------------------------------------
        evaluation = self.evaluator.evaluate(
            query=query,
            plan=initial_plan,
        )

        # --------------------------------------------------
        # Step 3: If the plan is already valid, keep it.
        #
        # Otherwise, ask the Revision Agent to reconsider
        # the plan using the evaluator's feedback.
        # --------------------------------------------------
        if evaluation.valid:
            final_plan = initial_plan
            revised = False

        else:
            final_plan = self.revision_agent.revise(
                query=query,
                original_plan=initial_plan,
                evaluation=evaluation,
            )

            revised = True

        # --------------------------------------------------
        # Keep intermediate outputs.
        #
        # This will be useful later for:
        # - debugging
        # - evaluation
        # - measuring critic effectiveness
        # - explaining the agent workflow in experiments
        # --------------------------------------------------
        return {
            "query": query,
            "initial_plan": initial_plan,
            "evaluation": evaluation,
            "final_plan": final_plan,
            "revised": revised,
        }