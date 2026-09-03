from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import Transaction
from datetime import datetime, timedelta

router = APIRouter()

@router.post("/razorpay")
def razorpay_webhook(payload: dict):
    # Mocking actual razorpay webhook ingestion
    print("Received Razorpay Webhook:", payload)
    return {"status": "ok"}

@router.post("/promise-to-pay/{transaction_id}")
def promise_to_pay(transaction_id: str, db: Session = Depends(get_db)):
    """Simulate a customer replying with 'I will pay tomorrow' on WhatsApp"""
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if txn:
        # Customer promises to pay tomorrow
        txn.promise_to_pay_date = datetime.now() + timedelta(days=1)
        db.commit()
        return {"status": "Promise logged", "date": txn.promise_to_pay_date}
    return {"error": "Transaction not found"}
