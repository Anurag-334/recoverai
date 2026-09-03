from app.core.policies import (
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
)


def test_successful_payment_is_blocked():

    policy = RecoveryPolicy()

    context = RecoveryContext(
        transaction_id="TXN_001",
        amount=5000,
        recovery_probability=0.90,
        failure_reason="bank_timeout",
        event_type="payment_failed",
        attempt_number=1,
        messages_sent=0,
        payment_succeeded=True,
    )

    decision = policy.evaluate(
        context,
        RecoveryAction.RETRY_PAYMENT,
    )

    assert decision.allowed is False
    assert decision.action == RecoveryAction.DO_NOTHING


def test_retry_limit_is_enforced():

    policy = RecoveryPolicy()

    context = RecoveryContext(
        transaction_id="TXN_002",
        amount=5000,
        recovery_probability=0.90,
        failure_reason="bank_timeout",
        event_type="payment_failed",
        attempt_number=2,
        messages_sent=0,
    )

    decision = policy.evaluate(
        context,
        RecoveryAction.RETRY_PAYMENT,
    )

    assert decision.allowed is False
    assert decision.action == RecoveryAction.ESCALATE_TO_MERCHANT


def test_low_value_negative_ev_is_blocked():

    policy = RecoveryPolicy()

    context = RecoveryContext(
        transaction_id="TXN_003",
        amount=100,
        recovery_probability=0.01,
        failure_reason="unknown",
        event_type="payment_failed",
        attempt_number=1,
        messages_sent=0,
    )

    decision = policy.evaluate(
        context,
        RecoveryAction.RETRY_PAYMENT,
    )

    assert decision.allowed is False
    assert decision.action == RecoveryAction.DO_NOTHING


def test_high_value_retry_requires_approval():

    policy = RecoveryPolicy()

    context = RecoveryContext(
        transaction_id="TXN_004",
        amount=100_000,
        recovery_probability=0.90,
        failure_reason="bank_timeout",
        event_type="payment_failed",
        attempt_number=1,
        messages_sent=0,
    )

    decision = policy.evaluate(
        context,
        RecoveryAction.RETRY_PAYMENT,
    )

    assert decision.allowed is False
    assert decision.action == RecoveryAction.ESCALATE_TO_MERCHANT