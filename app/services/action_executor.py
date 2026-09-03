import razorpay
from app.core.config import settings
from app.core.policies import RecoveryAction

class ActionExecutor:
    def __init__(self):
        self.rzp = None
        if settings.razorpay_key_id and settings.razorpay_key_secret:
            self.rzp = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))

    def execute(self, action: str, transaction_id: str) -> str:
        """
        Execute the recovery action using Razorpay Test Mode or Mock.
        """
        print(f"[ActionExecutor] Executing {action} for {transaction_id}")
        
        if action == RecoveryAction.RETRY_PAYMENT.value:
            return self._create_payment_link(transaction_id)
        elif action == RecoveryAction.SEND_PAYMENT_REMINDER.value:
            return f"SUCCESS: WhatsApp payment reminder sent for {transaction_id}."
        elif action == RecoveryAction.SEND_CHECKOUT_REMINDER.value:
            return f"SUCCESS: WhatsApp checkout reminder sent for {transaction_id}."
        elif action == RecoveryAction.ESCALATE_TO_MERCHANT.value:
            return f"SUCCESS: Escalated to merchant support queue for {transaction_id}."
        elif action == RecoveryAction.DO_NOTHING.value:
            return "SUCCESS: No action taken."
        else:
            return f"ERROR: Unknown action {action}."

    def _create_payment_link(self, transaction_id: str) -> str:
        if self.rzp:
            try:
                # Real Razorpay integration
                payment_link = self.rzp.payment_link.create({
                    "amount": 1000, # Demo default in paise
                    "currency": "INR",
                    "description": f"Recovery for {transaction_id}",
                    "reference_id": transaction_id,
                    "reminder_enable": True
                })
                url = payment_link.get('short_url', 'Unknown URL')
                return f"SUCCESS: Razorpay Payment Link Generated -> {url}"
            except Exception as e:
                return f"ERROR generating Razorpay link: {str(e)}"
        else:
            # Fallback mock for Demo without keys
            return f"SUCCESS: Mock Payment Link Generated -> https://rzp.io/i/mock_{transaction_id}"