import pandas as pd
from sqlalchemy.orm import Session
from app.database.models import Transaction
import random

class RazorpayMockService:
    def __init__(self, data_path="data/synthetic/recovery_events.csv"):
        self.data_path = data_path

    def seed_database(self, db: Session, limit: int = 50):
        """Seed the database with a subset of synthetic transactions for the demo."""
        # clear existing
        db.query(Transaction).delete()
        db.commit()
        
        df = pd.read_csv(self.data_path)
        # sample some failed transactions
        failed_df = df[df["recovered"] == 0].sample(n=limit, random_state=42)
        
        for _, row in failed_df.iterrows():
            txn = Transaction(
                transaction_id=row["transaction_id"],
                customer_id=row["customer_id"],
                amount=row["amount"],
                payment_method=row["payment_method"],
                event_type=row["event_type"],
                failure_reason=row["failure_reason"],
                attempt_number=row["attempt_number"],
                customer_segment=row["customer_segment"],
                days_since_failure=row["days_since_failure"],
                previous_success_rate=row["previous_success_rate"],
                customer_lifetime_value=row["customer_lifetime_value"],
                is_subscription=bool(row["is_subscription"]),
                invoice_age=row["invoice_age"],
                recovered=False,
                language_preference=random.choice(["Hinglish", "Hindi", "Tamil", "English"])
            )
            db.add(txn)
        
        db.commit()
        print(f"Seeded {limit} transactions into database.")
