from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import Transaction, AuditLog, BanditEvent
from app.agents.negotiation_agent import NegotiationAgent
from app.core.bandit import bandit
from datetime import datetime, timedelta, timezone
import json
import numpy as np

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
    
    # Process Actions & determine partial bandit reward
    partial_reward = None
    if result.intent == "promise_to_pay":
        txn.promise_to_pay_date = datetime.now() + timedelta(days=1)
        partial_reward = 0.3
    elif result.intent == "request_split":
        txn.split_payment_active = True
        partial_reward = 0.5
        
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
    
    # Record partial reward for the bandit if negotiation progressed
    if partial_reward is not None:
        _record_bandit_partial_reward(db, transaction_id, partial_reward)
    
    return {
        "status": "success", 
        "intent": result.intent, 
        "reply": result.reply_message
    }


def _record_bandit_partial_reward(
    db: Session,
    transaction_id: str,
    reward: float,
):
    """
    Find the most recent unrewarded bandit event for this transaction
    and record a partial reward from negotiation progress.
    """
    event = (
        db.query(BanditEvent)
        .filter(
            BanditEvent.transaction_id == transaction_id,
            BanditEvent.reward.is_(None),
        )
        .order_by(BanditEvent.timestamp.desc())
        .first()
    )

    if not event or not event.context_vector:
        return

    event.reward = reward
    event.reward_observed_at = datetime.now(timezone.utc)
    db.commit()

    # Update the bandit model weights
    context_vector = np.array(
        json.loads(event.context_vector), dtype=np.float64
    )
    bandit.update_reward(event.arm_selected, context_vector, reward)
