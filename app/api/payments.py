from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.database import get_db
from app.database.models import Transaction, AuditLog

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
