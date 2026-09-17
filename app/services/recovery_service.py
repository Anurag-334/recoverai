import json

from sqlalchemy.orm import Session
from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.message_agent import MessageAgent
from app.core.policies import RecoveryAction, RecoveryContext, RecoveryPolicy
from app.core.audit import create_audit_log
from app.core.bandit import bandit, extract_context_vector
from app.ml.predictor import predictor
from app.services.action_executor import ActionExecutor
from app.database.models import Transaction, BanditEvent

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

        # 1. Predict Recovery Probability (XGBoost)
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

        # 2. Build Recovery Context
        context = RecoveryContext(
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            recovery_probability=recovery_prob,
            failure_reason=txn.failure_reason,
            event_type=txn.event_type,
            attempt_number=txn.attempt_number,
            messages_sent=0,  # Simplifying for now
            payment_succeeded=txn.recovered,
            previous_success_rate=txn.previous_success_rate,
            customer_segment=txn.customer_segment,
            language_preference=txn.language_preference
        )

        # 3. 🎰 Contextual Bandit selects the best arm
        ctx_vector = extract_context_vector(context)
        arm_selection = bandit.select_arm(ctx_vector)
        bandit_action = RecoveryAction(arm_selection.arm)

        # 4. AI Diagnosis (advisory — used for reasoning/message, not action)
        diagnosis = self.agent.diagnose(context)

        # 5. Policy Safety Check (on the bandit's chosen arm)
        decision = self.policy.evaluate(context=context, proposed_action=bandit_action)

        # 6. Generate Message if needed
        generated_message = None
        if decision.allowed and decision.action in (
            RecoveryAction.SEND_PAYMENT_REMINDER,
            RecoveryAction.SEND_CHECKOUT_REMINDER,
        ):
            generated_message = self.message_agent.generate_message(context)

        # 7. Execute Action
        execution_status = None
        if decision.allowed:
            execution_status = self.executor.execute(decision.action.value, txn.transaction_id)
        else:
            execution_status = "BLOCKED BY POLICY"
            
        # 8. Update transaction if successful recovery action
        reward = None
        if decision.action == RecoveryAction.RETRY_PAYMENT and decision.allowed:
            txn.recovered = True  # For demo purpose
            reward = 1.0
            db.commit()

        # 9. Record BanditEvent for learning
        bandit_event = BanditEvent(
            transaction_id=txn.transaction_id,
            context_vector=json.dumps(arm_selection.context_vector),
            arm_selected=arm_selection.arm,
            arm_scores=json.dumps(arm_selection.scores),
            exploration_score=arm_selection.exploration_score,
            reward=reward,
        )
        db.add(bandit_event)
        db.commit()

        # 10. If we already know the reward, update bandit immediately
        if reward is not None:
            bandit.update_reward(arm_selection.arm, ctx_vector, reward)

        # 11. Audit Logging (includes bandit decision)
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
            execution_status=execution_status,
            bandit_arm_selected=arm_selection.arm,
            bandit_exploration_score=arm_selection.exploration_score,
        )

        return {
            "transaction_id": txn.transaction_id,
            "recovery_probability": recovery_prob,
            "diagnosis": diagnosis.dict(),
            "bandit_decision": {
                "arm_selected": arm_selection.arm,
                "scores": arm_selection.scores,
                "exploration_score": arm_selection.exploration_score,
            },
            "policy_decision": {
                "action": decision.action.value,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "expected_value": decision.expected_value
            },
            "execution_status": execution_status,
            "audit_log_id": log.id
        }