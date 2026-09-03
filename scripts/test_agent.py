from dotenv import load_dotenv

from app.agents.diagnosis_agent import DiagnosisAgent
from app.core.policies import RecoveryContext

load_dotenv()


def main():

    context = RecoveryContext(
        transaction_id="TXN_ALREAD_PAID",
        amount=5000,
        recovery_probability=0.95,
        failure_reason="bank_timeout",
        event_type="payment_failed",
        attempt_number=1,
        messages_sent=0,
        payment_succeeded=True,
    )

    agent = DiagnosisAgent()

    result = agent.diagnose(context)

    print("\nDIAGNOSIS")
    print(result.diagnosis)

    print("\nRECOMMENDED ACTION")
    print(result.recommended_action)

    print("\nCONFIDENCE")
    print(result.confidence)

    print("\nREASONING")
    print(result.reasoning)

    print("\nCUSTOMER MESSAGE")
    print(result.customer_message)


if __name__ == "__main__":
    main()