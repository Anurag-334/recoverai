from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database.database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    customer_id = Column(String, index=True)
    amount = Column(Float)
    payment_method = Column(String)
    event_type = Column(String)
    failure_reason = Column(String)
    attempt_number = Column(Integer)
    customer_segment = Column(String)
    days_since_failure = Column(Integer)
    previous_success_rate = Column(Float)
    customer_lifetime_value = Column(Float)
    is_subscription = Column(Boolean)
    invoice_age = Column(Integer)
    recovered = Column(Boolean, default=False)
    
    # Promise to pay tracker
    promise_to_pay_date = Column(DateTime, nullable=True)
    promise_kept = Column(Boolean, nullable=True)
    
    # V3 Upgrades
    language_preference = Column(String, default="Hinglish")
    split_payment_active = Column(Boolean, default=False)
    negotiation_history = Column(Text, nullable=True) # JSON string
    
    # Audit trail relationship
    audit_logs = relationship("AuditLog", back_populates="transaction")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # ML
    recovery_probability = Column(Float, nullable=True)
    
    # AI Diagnosis
    agent_diagnosis = Column(Text, nullable=True)
    agent_recommended_action = Column(String, nullable=True)
    agent_reasoning = Column(Text, nullable=True)
    agent_confidence = Column(Float, nullable=True)
    generated_message = Column(Text, nullable=True)
    
    # Policy Decision
    policy_allowed = Column(Boolean, nullable=True)
    policy_reason = Column(Text, nullable=True)
    policy_expected_value = Column(Float, nullable=True)
    
    # Execution
    final_action = Column(String, nullable=True)
    execution_status = Column(String, nullable=True)
    
    transaction = relationship("Transaction", back_populates="audit_logs")
