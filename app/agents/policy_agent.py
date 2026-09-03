from app.core.policies import (
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
)


class PolicyAgent:

    def __init__(self):
        self.policy = RecoveryPolicy()

    def evaluate(
        self,
        context: RecoveryContext,
        proposed_action: RecoveryAction,
    ):
        """
        Deterministic safety gate.

        The AI may recommend an action, but this layer
        decides whether that action is actually allowed.
        """

        return self.policy.evaluate(
            context=context,
            proposed_action=proposed_action,
        )