from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import Transaction, AuditLog
from app.agents.negotiation_agent import NegotiationAgent
from datetime import datetime, timedelta
import json

router = APIRouter()

@router.post("/razorpay")
def razorpay_webhook(payload: dict):
    # Mocking actual razorpay webhook ingestion
    print("Received Razorpay Webhook:", payload)
    return {"status": "ok"}

@router.post("/whatsapp-reply/{transaction_id}")
def whatsapp_reply(transaction_id: str, payload: dict, db: Session = Depends(get_db)):
    """Simulate a customer replying on WhatsApp"""
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not txn:
        return {"error": "Transaction not found"}
        
    customer_message = payload.get("message", "")
    
    # Run Negotiation Agent
    agent = NegotiationAgent()
    result = agent.negotiate(txn.transaction_id, txn.amount, customer_message)
    
    # Process Actions
    if result.intent == "promise_to_pay":
        txn.promise_to_pay_date = datetime.now() + timedelta(days=1)
    elif result.intent == "request_split":
        txn.split_payment_active = True
        
    # Append to History
    history = json.loads(txn.negotiation_history) if txn.negotiation_history else []
    history.append({"role": "customer", "content": customer_message})
    history.append({"role": "agent", "content": result.reply_message})
    txn.negotiation_history = json.dumps(history)
    
    # Create Audit Log
    log = AuditLog(
        transaction_id=txn.transaction_id,
        recovery_probability=txn.previous_success_rate or 0.85,
        agent_diagnosis=f"Customer Intent: {result.intent}",
        agent_recommended_action=result.action_taken,
        agent_reasoning="Negotiation Agent processed WhatsApp reply.",
        agent_confidence=0.95,
        policy_allowed=True,
        policy_reason="Negotiation response approved by policy.",
        policy_expected_value=float(txn.amount),
        generated_message=result.reply_message,
        execution_status="SUCCESS: WhatsApp reply sent.",
        final_action="negotiation_reply"
    )
    db.add(log)
    db.commit()
    
    return {
        "status": "success", 
        "intent": result.intent, 
        "reply": result.reply_message
    }
