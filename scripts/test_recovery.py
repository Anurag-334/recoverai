from app.core.policies import RecoveryContext
from app.services.recovery_service import RecoveryService


def main():

    service = RecoveryService()

    context = RecoveryContext(
        transaction_id="TXN_DEMO_001",
        amount=1000000,
        recovery_probability=0.87,
        failure_reason="bank_timeout",
        event_type="payment_failed",
        attempt_number=1,
        messages_sent=0,
    )

    decision = service.decide(context)

    print("ACTION")
    print(decision.action)

    print("\nALLOWED")
    print(decision.allowed)

    print("\nREASON")
    print(decision.reason)

    print("\nEXPECTED VALUE")
    print(f"₹{decision.expected_value:,.2f}")

    print("\nACTION COST")
    print(f"₹{decision.action_cost:,.2f}")


if __name__ == "__main__":
    main()