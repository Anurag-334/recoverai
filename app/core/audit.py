from sqlalchemy.orm import Session
from app.database.models import AuditLog
from typing import Optional

def create_audit_log(
    db: Session,
    transaction_id: str,
    recovery_probability: Optional[float] = None,
    agent_diagnosis: Optional[str] = None,
    agent_recommended_action: Optional[str] = None,
    agent_reasoning: Optional[str] = None,
    agent_confidence: Optional[float] = None,
    generated_message: Optional[str] = None,
    policy_allowed: Optional[bool] = None,
    policy_reason: Optional[str] = None,
    policy_expected_value: Optional[float] = None,
    final_action: Optional[str] = None,
    execution_status: Optional[str] = None
) -> AuditLog:
    
    log = AuditLog(
        transaction_id=transaction_id,
        recovery_probability=recovery_probability,
        agent_diagnosis=agent_diagnosis,
        agent_recommended_action=agent_recommended_action,
        agent_reasoning=agent_reasoning,
        agent_confidence=agent_confidence,
        generated_message=generated_message,
        policy_allowed=policy_allowed,
        policy_reason=policy_reason,
        policy_expected_value=policy_expected_value,
        final_action=final_action,
        execution_status=execution_status
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
