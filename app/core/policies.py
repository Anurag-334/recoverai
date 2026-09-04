from dataclasses import dataclass
from enum import Enum


class RecoveryAction(str, Enum):
    RETRY_PAYMENT = "retry_payment"
    SEND_PAYMENT_REMINDER = "send_payment_reminder"
    SEND_CHECKOUT_REMINDER = "send_checkout_reminder"
    ESCALATE_TO_MERCHANT = "escalate_to_merchant"
    DO_NOTHING = "do_nothing"


@dataclass
class RecoveryContext:
    transaction_id: str
    amount: float
    recovery_probability: float

    failure_reason: str
    event_type: str

    attempt_number: int
    messages_sent: int

    payment_succeeded: bool = False
    previous_success_rate: float = 0.85
    customer_segment: str = "standard"
    language_preference: str = "Hinglish"


@dataclass
class PolicyDecision:
    action: RecoveryAction
    allowed: bool
    reason: str
    expected_value: float
    action_cost: float


class RecoveryPolicy:
    """
    Deterministic safety and decision layer for RecoverAI.

    The LLM/agent may recommend an action, but this
    policy determines whether that action is permitted.
    """

    MAX_RETRIES = 2
    MAX_MESSAGES = 2

    HIGH_VALUE_THRESHOLD = 50_000

    RETRY_COST = 10.0
    MESSAGE_COST = 2.0
    ESCALATION_COST = 25.0

    MIN_EXPECTED_VALUE = 0.0

    def calculate_expected_value(
        self,
        amount: float,
        recovery_probability: float,
        action_cost: float,
    ) -> float:
        expected_revenue = amount * recovery_probability

        return expected_revenue - action_cost

    def evaluate(
        self,
        context: RecoveryContext,
        proposed_action: RecoveryAction,
    ) -> PolicyDecision:

        # --------------------------------------------------
        # Rule 1: Never act on an already successful payment
        # --------------------------------------------------

        if context.payment_succeeded:
            return PolicyDecision(
                action=RecoveryAction.DO_NOTHING,
                allowed=False,
                reason="Payment has already succeeded.",
                expected_value=0.0,
                action_cost=0.0,
            )

        # --------------------------------------------------
        # Rule 2: Retry limit
        # --------------------------------------------------

        if (
            proposed_action == RecoveryAction.RETRY_PAYMENT
            and context.attempt_number >= self.MAX_RETRIES
        ):
            return PolicyDecision(
                action=RecoveryAction.ESCALATE_TO_MERCHANT,
                allowed=False,
                reason=(
                    f"Retry limit reached: "
                    f"{context.attempt_number} attempts."
                ),
                expected_value=0.0,
                action_cost=self.ESCALATION_COST,
            )

        # --------------------------------------------------
        # Rule 3: Message limit
        # --------------------------------------------------

        message_actions = {
            RecoveryAction.SEND_PAYMENT_REMINDER,
            RecoveryAction.SEND_CHECKOUT_REMINDER,
        }

        if (
            proposed_action in message_actions
            and context.messages_sent >= self.MAX_MESSAGES
        ):
            return PolicyDecision(
                action=RecoveryAction.ESCALATE_TO_MERCHANT,
                allowed=False,
                reason=(
                    f"Customer communication limit reached: "
                    f"{context.messages_sent} messages."
                ),
                expected_value=0.0,
                action_cost=self.ESCALATION_COST,
            )

        # --------------------------------------------------
        # Determine action cost
        # --------------------------------------------------

        if proposed_action == RecoveryAction.RETRY_PAYMENT:
            action_cost = self.RETRY_COST

        elif proposed_action in message_actions:
            action_cost = self.MESSAGE_COST

        elif proposed_action == RecoveryAction.ESCALATE_TO_MERCHANT:
            action_cost = self.ESCALATION_COST

        else:
            action_cost = 0.0

        # --------------------------------------------------
        # Expected value
        # --------------------------------------------------

        expected_value = self.calculate_expected_value(
            amount=context.amount,
            recovery_probability=context.recovery_probability,
            action_cost=action_cost,
        )

        # --------------------------------------------------
        # Rule 4: Don't spend money on negative EV actions
        # --------------------------------------------------

        if (
            proposed_action != RecoveryAction.ESCALATE_TO_MERCHANT
            and expected_value <= self.MIN_EXPECTED_VALUE
        ):
            return PolicyDecision(
                action=RecoveryAction.DO_NOTHING,
                allowed=False,
                reason=(
                    "Expected recovery value does not justify "
                    "the action cost."
                ),
                expected_value=expected_value,
                action_cost=action_cost,
            )

        # --------------------------------------------------
        # Rule 5: High-value transaction gate
        # --------------------------------------------------

        if (
            context.amount >= self.HIGH_VALUE_THRESHOLD
            and proposed_action == RecoveryAction.RETRY_PAYMENT
        ):
            return PolicyDecision(
                action=RecoveryAction.ESCALATE_TO_MERCHANT,
                allowed=False,
                reason=(
                    "High-value payment requires merchant approval "
                    "before retry."
                ),
                expected_value=expected_value,
                action_cost=self.ESCALATION_COST,
            )

        # --------------------------------------------------
        # Action approved
        # --------------------------------------------------

        return PolicyDecision(
            action=proposed_action,
            allowed=True,
            reason="Action passed all policy checks.",
            expected_value=expected_value,
            action_cost=action_cost,
        )