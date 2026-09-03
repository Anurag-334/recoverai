from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.policy_agent import PolicyAgent
from app.core.policies import RecoveryAction


class RecoveryAgent:

    def __init__(self):
        self.diagnosis_agent = DiagnosisAgent()
        self.policy_agent = PolicyAgent()

    def run(self, context):
        """
        Complete AI recovery decision pipeline.

        1. Diagnose the payment problem.
        2. Ask the AI for a recommended action.
        3. Pass that recommendation through the
           deterministic policy gate.
        """

        # --------------------------------------------
        # Step 1: AI diagnosis
        # --------------------------------------------

        diagnosis = self.diagnosis_agent.diagnose(context)

        # --------------------------------------------
        # Step 2: Convert AI recommendation into
        #         our internal RecoveryAction enum
        # --------------------------------------------

        proposed_action = RecoveryAction(
            diagnosis.recommended_action.value
        )

        # --------------------------------------------
        # Step 3: Safety / policy evaluation
        # --------------------------------------------

        policy_decision = self.policy_agent.evaluate(
            context=context,
            proposed_action=proposed_action,
        )

        return {
            "diagnosis": diagnosis,
            "proposed_action": proposed_action,
            "policy_decision": policy_decision,
        }