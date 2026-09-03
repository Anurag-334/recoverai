from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.agents.schemas import DiagnosisResult
from app.core.config import settings


class DiagnosisAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

        self.structured_llm = self.llm.with_structured_output(
            DiagnosisResult
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the diagnosis component of RecoverAI,
an autonomous revenue recovery system.

Your job is to analyze a failed or at-risk payment
and recommend the most appropriate recovery action.

IMPORTANT SAFETY RULES:

1. Never assume you can execute an action.
2. Your output is only a recommendation.
3. A deterministic policy engine will decide whether
   the recommendation is allowed.
4. Never recommend retrying a payment that has already
   succeeded.
5. Consider payment failure reason, attempt count,
   customer history, transaction amount and recovery
   probability.
6. Avoid unnecessary customer contact.
7. Be concise and explain your reasoning.

Available actions:

- retry_payment
- send_payment_reminder
- send_checkout_reminder
- escalate_to_merchant
- do_nothing

Return structured output only.
""",
                ),
                (
                    "human",
                    """
Transaction information:

Transaction ID: {transaction_id}
Amount: ₹{amount}

Event type: {event_type}
Failure reason: {failure_reason}

Attempt number: {attempt_number}
Messages sent: {messages_sent}

Previous success rate: {previous_success_rate}
Customer segment: {customer_segment}

Recovery probability from ML model:
{recovery_probability}

Has payment already succeeded:
{payment_succeeded}
""",
                ),
            ]
        )

    def diagnose(self, context) -> DiagnosisResult:

        chain = self.prompt | self.structured_llm

        result = chain.invoke(
            {
                "transaction_id": context.transaction_id,
                "amount": context.amount,
                "event_type": context.event_type,
                "failure_reason": context.failure_reason,
                "attempt_number": context.attempt_number,
                "messages_sent": context.messages_sent,
                "previous_success_rate": context.previous_success_rate,
                "customer_segment": context.customer_segment,
                "recovery_probability": context.recovery_probability,
                "payment_succeeded": context.payment_succeeded,
            }
        )

        return result