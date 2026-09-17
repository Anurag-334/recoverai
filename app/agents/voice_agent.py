import json
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from app.core.config import settings

class VoiceResponse(BaseModel):
    spoken_reply: str = Field(description="The verbal response (natural conversational speech, NO markdown, NO URLs in spoken text, 1-3 sentences max)")
    intent: Literal['promise_to_pay', 'request_split', 'pay_now', 'dispute_or_query', 'greeting', 'decline', 'unknown'] = Field(description="The detected intent of the customer")
    action_taken: str = Field(description="Internal description of action taken")
    suggested_responses: List[str] = Field(description="3-4 quick reply chips for the UI (e.g. 'I can pay half now', 'I'll pay tomorrow', 'Why did it fail?', 'Send me the link')")
    notification: Optional[Dict[str, str]] = Field(default=None, description="If an action triggers a notification, include {title, body, type}")

class VoiceAgent:
    def __init__(self):
        self.api_key = settings.groq_api_key
        self.model_name = settings.llm_model or "llama3-70b-8192"
        self.llm = None
        
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=self.api_key,
                    model_name=self.model_name,
                    max_tokens=500
                ).with_structured_output(VoiceResponse)
            except Exception as e:
                print(f"Error initializing ChatGroq: {e}")
                self.llm = None

    def _get_system_prompt(self, transaction_id: str, amount: float, language: str) -> str:
        return f"""You are an empathetic, highly professional customer care representative for RecoverAI, calling on behalf of a merchant regarding transaction {transaction_id}.
Your goal is to politely assist the customer in resolving an unsuccessful payment of {amount}.

KEY BEHAVIORS:
1. Adopt a localized Customer Care persona:
   - For English: Your name is 'Sarah'. Greet with 'Hello sir/ma'am'.
   - For Hindi/Hinglish: Your name is 'Priya'. Greet with 'Namaste sir/ma'am'.
   - For Tamil: Your name is 'Kavya'. Greet with 'Vanakkam sir/ma'am'.
2. Always be extremely polite and empathetic. Use phrases like 'I completely understand', 'I apologize for the inconvenience', and 'Let me help you with that'.
3. SPEAK NATURALLY as if on a phone call. Keep responses to 1-3 short sentences.
4. NEVER output markdown, bullet points, or URLs. Say "I will send you a secure link via SMS".
5. **SPECIAL AUTHORITY**: You have the managerial authority to WAIVE LATE FEES. If the customer hesitates, complains about money, or refuses to pay, proactively offer to waive all late fees if they complete the payment or split payment today.

Match the customer's language preference: {language}."""

    def generate_greeting(self, transaction_id: str, amount: float, failure_reason: str, customer_name: str, language: str) -> VoiceResponse:
        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(self._get_system_prompt(transaction_id, amount, language)),
                    HumanMessagePromptTemplate.from_template("Generate an initial phone greeting for {customer_name}. The previous payment failed due to {failure_reason}.")
                ])
                chain = prompt | self.llm
                response = chain.invoke({
                    "customer_name": customer_name,
                    "failure_reason": failure_reason
                })
                return response
            except Exception as e:
                print(f"LLM fallback triggered for greeting: {e}")
                return self._fallback_greeting(transaction_id, amount, failure_reason, customer_name, language)
        else:
            return self._fallback_greeting(transaction_id, amount, failure_reason, customer_name, language)

    def process_turn(self, transaction_id: str, amount: float, customer_message: str, conversation_history: List[Dict[str, str]], language: str) -> VoiceResponse:
        if self.llm:
            try:
                messages = [("system", self._get_system_prompt(transaction_id, amount, language))]
                for msg in conversation_history:
                    if msg['role'] == 'assistant':
                        messages.append(("assistant", msg['content']))
                    elif msg['role'] == 'user':
                        messages.append(("user", msg['content']))
                
                messages.append(("user", customer_message))
                
                response = self.llm.invoke(messages)
                return response
            except Exception as e:
                print(f"LLM fallback triggered for process_turn: {e}")
                return self._fallback_process_turn(transaction_id, amount, customer_message, language)
        else:
            return self._fallback_process_turn(transaction_id, amount, customer_message, language)
            
    def generate_summary(self, transaction_id: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        user_msgs = [m for m in conversation_history if m['role'] == 'user']
        final_intent = "unknown"
        
        if user_msgs:
            last_msg = user_msgs[-1]['content'].lower()
            if any(k in last_msg for k in ['half', 'split', 'partial', 'installment', 'aadha']):
                final_intent = 'request_split'
            elif any(k in last_msg for k in ['tomorrow', 'later', 'next week', 'kal', 'baad']):
                final_intent = 'promise_to_pay'
            elif any(k in last_msg for k in ['pay now', 'link', 'abhi']):
                final_intent = 'pay_now'
        
        return {
            "duration_turns": len(conversation_history) // 2,
            "final_intent": final_intent,
            "actions_taken": ["Call completed"],
            "transcript": conversation_history
        }

    def _fallback_greeting(self, transaction_id: str, amount: float, failure_reason: str, customer_name: str, language: str) -> VoiceResponse:
        if 'hi' in language.lower() or 'hin' in language.lower():
            spoken = f"Namaste sir, main RecoverAI customer care se Priya baat kar rahi hoon. Aapka {amount} ka payment fail ho gaya tha bank issue ki wajah se. Main aapki late fees maaf kar sakti hoon agar hum isko abhi settle kar lein. Kya main madad karoon?"
        elif 'ta' in language.lower() or 'tamil' in language.lower():
            spoken = f"Vanakkam sir, naan RecoverAI-la irundhu Kavya pesuren. Unga {amount} payment unsuccessful aagirukku. Namma inaikku idhe settle panna late fees waive panna mudiyum. Naan epdi help pannatum?"
        else:
            spoken = f"Hello sir, this is Sarah from RecoverAI customer care. I'm calling about a recent unsuccessful payment of {amount} due to {failure_reason}. I can waive your late fees if we resolve this today. How can I assist you?"
            
        return VoiceResponse(
            spoken_reply=spoken,
            intent="greeting",
            action_taken="Initiated call",
            suggested_responses=["Yes, tell me more", "Why did it fail?", "I'll pay later", "I can't pay now"],
            notification=None
        )

    def _fallback_process_turn(self, transaction_id: str, amount: float, customer_message: str, language: str) -> VoiceResponse:
        msg = customer_message.lower()
        
        if any(k in msg for k in ['half', 'split', 'partial', 'installment', 'aadha']):
            return VoiceResponse(
                spoken_reply="I can help you split this payment into two installments. I'll send you a link via SMS right now.",
                intent="request_split",
                action_taken="Provided split payment option",
                suggested_responses=["Send the link", "No, full payment", "How does it work?"],
                notification={"title": "Split Payment Activated", "body": f"Link sent for split payment of {amount}.", "type": "split_payment"}
            )
        elif any(k in msg for k in ['tomorrow', 'later', 'next week', 'kal', 'baad mein', 'baad']):
            return VoiceResponse(
                spoken_reply="No problem, I have logged your promise to pay later. Please ensure it's completed by tomorrow to avoid late fees.",
                intent="promise_to_pay",
                action_taken="Logged promise to pay",
                suggested_responses=["Okay", "Can I have more time?", "Send reminder"],
                notification={"title": "Promise Logged", "body": "Customer promised to pay tomorrow.", "type": "promise_logged"}
            )
        elif any(k in msg for k in ['pay now', 'send link', 'send me', 'link bhejo', 'abhi', 'link please', 'the link']):
            return VoiceResponse(
                spoken_reply="Great, I'll send you a secure payment link via SMS immediately. You can use it to pay now.",
                intent="pay_now",
                action_taken="Sent payment link",
                suggested_responses=["Thanks", "Didn't receive it", "What are the payment methods?"],
                notification={"title": "Payment Link Sent", "body": "Secure payment link sent to customer.", "type": "payment_link"}
            )
        elif any(k in msg for k in ['why', 'fail', 'kyu', 'reason', 'problem']):
            return VoiceResponse(
                spoken_reply="Your payment failed due to an issue with your bank or insufficient funds. Would you like to try another payment method?",
                intent="dispute_or_query",
                action_taken="Explained failure reason",
                suggested_responses=["I'll try again", "Send me link", "Let me check my bank"],
                notification=None
            )
        elif any(k in msg for k in ['how', 'what', 'didn\'t receive', 'not receive', 'help', 'mean']):
            return VoiceResponse(
                spoken_reply="Let me explain. You can pay via UPI or Card using the link I send, or we can split the payment into installments. Which do you prefer?",
                intent="unknown",
                action_taken="Clarified process",
                suggested_responses=["Send me link", "Split payment", "I'll pay tomorrow"],
                notification=None
            )
        elif any(k in msg for k in ['fee', 'late fee', 'waive', 'maaf', 'penalty', 'extra']):
            return VoiceResponse(
                spoken_reply="Yes sir! As a courtesy, I can completely waive the late fees today if you can clear the principal amount using the link I provide. Shall I send it?",
                intent="request_split",
                action_taken="Offered fee waiver",
                suggested_responses=["Yes, send link", "No, I can't pay", "How much is the fee?"],
                notification=None
            )
        elif any(k in msg for k in ["can't", "cannot", "nahi", "refuse", "don't want", "won't", "wont", "cancel"]):
            return VoiceResponse(
                spoken_reply="I understand. If you're facing difficulties, we can discuss a split payment plan. Otherwise, please contact support.",
                intent="decline",
                action_taken="Handled decline",
                suggested_responses=["Tell me about split payment", "Okay, bye", "I need more time"],
                notification=None
            )
        else:
            return VoiceResponse(
                spoken_reply="I'm sorry, I didn't quite catch that. Would you like me to send you a payment link so you can pay at your convenience?",
                intent="unknown",
                action_taken="Clarified intent",
                suggested_responses=["Yes, send link", "No, I want to split", "I'll pay later"],
                notification=None
            )
