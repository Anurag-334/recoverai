from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.core.config import settings

class MessageResult(BaseModel):
    message: str = Field(description="The generated message in Hinglish")

class MessageAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.3,
            api_key=settings.groq_api_key,
        )
        self.structured_llm = self.llm.with_structured_output(MessageResult)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are the communication agent for RecoverAI.
Your job is to generate a short, personalized recovery message for a customer whose payment failed or invoice is overdue.
Write the message in conversational Hinglish (a mix of Hindi and English, written in English script).
Example: "Hi Ravi, aapka payment complete nahi hua due to bank timeout. Sirf ek click se dobara pay kar sakte hain: [LINK]"

Rules:
1. Keep it under 2 sentences.
2. Be polite and helpful.
3. Always include "[LINK]" where the payment link should go.
4. Mention the reason for failure if it helps (e.g., bank timeout, insufficient funds).
"""),
            ("human", """
Transaction ID: {transaction_id}
Amount: ₹{amount}
Reason for failure: {failure_reason}
Event type: {event_type}
""")
        ])

    def generate_message(self, context) -> str:
        chain = self.prompt | self.structured_llm
        result = chain.invoke({
            "transaction_id": context.transaction_id,
            "amount": context.amount,
            "failure_reason": context.failure_reason,
            "event_type": context.event_type,
        })
        return result.message
