from sqlalchemy.orm import Session
from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.message_agent import MessageAgent
from app.core.policies import RecoveryAction, RecoveryContext, RecoveryPolicy
from app.core.audit import create_audit_log
from app.ml.predictor import predictor
from app.services.action_executor import ActionExecutor
from app.database.models import Transaction

class RecoveryService:
    def __init__(self):
        self.policy = RecoveryPolicy()
        self.agent = DiagnosisAgent()
        self.message_agent = MessageAgent()
        self.executor = ActionExecutor()

    def process_transaction(self, db: Session, transaction_id: str):
        txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
        if not txn:
            raise ValueError(f"Transaction {transaction_id} not found")

        # 1. Predict Recovery Probability
        context_dict = {
            "amount": txn.amount,
            "payment_method": txn.payment_method,
            "event_type": txn.event_type,
            "failure_reason": txn.failure_reason,
            "attempt_number": txn.attempt_number,
            "customer_segment": txn.customer_segment,
            "days_since_failure": txn.days_since_failure,
            "previous_success_rate": txn.previous_success_rate,
            "customer_lifetime_value": txn.customer_lifetime_value,
            "is_subscription": txn.is_subscription,
            "invoice_age": txn.invoice_age
        }
        recovery_prob = predictor.predict_probability(context_dict)

        # 2. Setup Context
        context = RecoveryContext(
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            recovery_probability=recovery_prob,
            failure_reason=txn.failure_reason,
            event_type=txn.event_type,
            attempt_number=txn.attempt_number,
            messages_sent=0, # Simplifying for now
            payment_succeeded=txn.recovered,
            previous_success_rate=txn.previous_success_rate,
            customer_segment=txn.customer_segment
        )

        # 3. AI Diagnosis
        diagnosis = self.agent.diagnose(context)
        proposed_action = RecoveryAction(diagnosis.recommended_action.value)

        # 4. Policy Check
        decision = self.policy.evaluate(context=context, proposed_action=proposed_action)

        # 5. Generate Message if needed
        generated_message = None
        if decision.allowed and decision.action in (RecoveryAction.SEND_PAYMENT_REMINDER, RecoveryAction.SEND_CHECKOUT_REMINDER):
            generated_message = self.message_agent.generate_message(context)

        # 6. Execute Action
        execution_status = None
        if decision.allowed:
            execution_status = self.executor.execute(decision.action.value, txn.transaction_id)
        else:
            execution_status = "BLOCKED BY POLICY"
            
        # Update transaction if successful recovery action mocked
        if decision.action == RecoveryAction.RETRY_PAYMENT and decision.allowed:
            txn.recovered = True # For demo purpose
            db.commit()

        # 7. Audit Logging
        log = create_audit_log(
            db=db,
            transaction_id=txn.transaction_id,
            recovery_probability=recovery_prob,
            agent_diagnosis=diagnosis.diagnosis,
            agent_recommended_action=diagnosis.recommended_action.value,
            agent_reasoning=diagnosis.reasoning,
            agent_confidence=diagnosis.confidence,
            generated_message=generated_message,
            policy_allowed=decision.allowed,
            policy_reason=decision.reason,
            policy_expected_value=decision.expected_value,
            final_action=decision.action.value,
            execution_status=execution_status
        )

        return {
            "transaction_id": txn.transaction_id,
            "recovery_probability": recovery_prob,
            "diagnosis": diagnosis.dict(),
            "policy_decision": {
                "action": decision.action.value,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "expected_value": decision.expected_value
            },
            "execution_status": execution_status,
            "audit_log_id": log.id
        }