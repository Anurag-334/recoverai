from enum import Enum

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    RETRY_PAYMENT = "retry_payment"
    SEND_PAYMENT_REMINDER = "send_payment_reminder"
    SEND_CHECKOUT_REMINDER = "send_checkout_reminder"
    ESCALATE_TO_MERCHANT = "escalate_to_merchant"
    DO_NOTHING = "do_nothing"


class DiagnosisResult(BaseModel):
    diagnosis: str = Field(
        description="Concise explanation of the likely payment problem."
    )

    recommended_action: AgentAction

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    customer_message: str | None = None

    reasoning: str = Field(
        description="Short explanation supporting the recommendation."
    )