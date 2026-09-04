from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.database import get_db
from app.database.models import Transaction, AuditLog

from datetime import datetime

router = APIRouter()

@router.get("/transactions")
def get_transactions(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    return txns

@router.get("/audit-logs/{transaction_id}")
def get_audit_logs(transaction_id: str, db: Session = Depends(get_db)):
    logs = db.query(AuditLog).filter(AuditLog.transaction_id == transaction_id).all()
    return logs

@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    revenue_at_risk = db.query(func.sum(Transaction.amount)).filter(Transaction.recovered == False).scalar() or 0
    revenue_recovered = db.query(func.sum(Transaction.amount)).filter(Transaction.recovered == True).scalar() or 0
    
    attempted_recoveries = db.query(Transaction).filter(Transaction.audit_logs.any()).count()
    successful_recoveries = db.query(Transaction).filter(Transaction.recovered == True).count()
    success_rate = (successful_recoveries / attempted_recoveries * 100) if attempted_recoveries > 0 else 0
    
    action_counts = db.query(AuditLog.final_action, func.count(AuditLog.id)).group_by(AuditLog.final_action).all()
    actions = {action: count for action, count in action_counts if action}
    
    return {
        "revenue_at_risk": revenue_at_risk,
        "revenue_recovered": revenue_recovered,
        "success_rate": success_rate,
        "actions": actions
    }

@router.get("/agent-actions-breakdown")
def get_agent_actions_breakdown(db: Session = Depends(get_db)):
    txns = db.query(Transaction).filter(Transaction.audit_logs.any()).all()
    breakdown_list = []
    
    for txn in txns:
        logs = sorted(txn.audit_logs, key=lambda l: l.timestamp or datetime.min, reverse=True)
        latest_log = logs[0] if logs else None
        
        category = "reminded"
        status_label = "Intervention Initiated"
        badge_color = "blue"
        
        if txn.recovered:
            category = "restored"
            status_label = "Payment Restored & Captured"
            badge_color = "green"
        elif txn.split_payment_active:
            category = "negotiating"
            status_label = "In Negotiation: Split Installments Active"
            badge_color = "purple"
        elif txn.promise_to_pay_date:
            category = "negotiating"
            promise_str = txn.promise_to_pay_date.strftime("%b %d, %Y") if txn.promise_to_pay_date else "Tomorrow"
            status_label = f"In Negotiation: Promised to Pay by {promise_str}"
            badge_color = "purple"
        elif latest_log and latest_log.final_action == "negotiation_reply":
            category = "negotiating"
            status_label = "In Negotiation: Customer Replied on WhatsApp"
            badge_color = "purple"
        elif latest_log and latest_log.final_action in ("send_payment_reminder", "send_checkout_reminder"):
            category = "reminded"
            status_label = "Reminder Sent (Awaiting Response)"
            badge_color = "blue"
        elif latest_log and latest_log.final_action == "escalate_to_merchant":
            category = "escalated"
            status_label = "Escalated to Merchant Support"
            badge_color = "red"
        elif latest_log and latest_log.policy_allowed is False:
            category = "escalated"
            status_label = f"Blocked by Policy: {latest_log.policy_reason or 'Risk policy guard'}"
            badge_color = "orange"
            
        breakdown_list.append({
            "transaction_id": txn.transaction_id,
            "customer_id": txn.customer_id,
            "amount": float(txn.amount),
            "failure_reason": txn.failure_reason,
            "language_preference": txn.language_preference,
            "recovered": txn.recovered,
            "category": category,
            "status_label": status_label,
            "badge_color": badge_color,
            "latest_action": latest_log.final_action if latest_log else None,
            "latest_message": latest_log.generated_message if latest_log else None,
            "agent_reasoning": latest_log.agent_reasoning if latest_log else None,
            "agent_diagnosis": latest_log.agent_diagnosis if latest_log else None,
            "execution_status": latest_log.execution_status if latest_log else None,
            "timestamp": latest_log.timestamp.isoformat() if latest_log and latest_log.timestamp else None,
            "total_actions": len(logs)
        })
        
    breakdown_list.sort(key=lambda x: x["timestamp"] or "", reverse=True)
    
    counts = {
        "all": len(breakdown_list),
        "restored": sum(1 for x in breakdown_list if x["category"] == "restored"),
        "negotiating": sum(1 for x in breakdown_list if x["category"] == "negotiating"),
        "reminded": sum(1 for x in breakdown_list if x["category"] == "reminded"),
        "escalated": sum(1 for x in breakdown_list if x["category"] == "escalated"),
    }
    
    return {
        "counts": counts,
        "actions": breakdown_list
    }
