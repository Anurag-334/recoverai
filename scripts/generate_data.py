from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_TRANSACTIONS = 10_000


def generate_data(n: int = N_TRANSACTIONS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    transaction_ids = [f"TXN_{i:06d}" for i in range(1, n + 1)]
    customer_ids = [f"CUST_{rng.integers(1, 3001):05d}" for _ in range(n)]

    amount = np.round(
        rng.lognormal(mean=np.log(1500), sigma=0.8, size=n),
        2,
    )
    amount = np.clip(amount, 100, 100_000)

    payment_method = rng.choice(
        ["upi", "card", "netbanking", "wallet"],
        size=n,
        p=[0.55, 0.25, 0.15, 0.05],
    )

    event_type = rng.choice(
        [
            "payment_failed",
            "checkout_abandoned",
            "subscription_failed",
            "invoice_overdue",
        ],
        size=n,
        p=[0.45, 0.25, 0.15, 0.15],
    )

    failure_reason = rng.choice(
        [
            "bank_timeout",
            "insufficient_funds",
            "network_error",
            "authentication_failed",
            "limit_exceeded",
            "customer_abandoned",
            "mandate_failed",
            "unknown",
        ],
        size=n,
        p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.15, 0.08, 0.12],
    )

    attempt_number = rng.choice(
        [1, 2, 3, 4],
        size=n,
        p=[0.55, 0.25, 0.15, 0.05],
    )

    customer_segment = rng.choice(
        ["new", "regular", "loyal", "high_value"],
        size=n,
        p=[0.25, 0.40, 0.25, 0.10],
    )

    days_since_failure = rng.integers(0, 15, size=n)

    previous_success_rate = np.round(
        rng.beta(8, 2, size=n),
        3,
    )

    customer_lifetime_value = np.round(
        rng.lognormal(mean=np.log(10_000), sigma=0.9, size=n),
        2,
    )

    is_subscription = (
        event_type == "subscription_failed"
    ).astype(int)

    invoice_age = np.where(
        event_type == "invoice_overdue",
        rng.integers(1, 61, size=n),
        0,
    )

    # ---------------------------------------------------------
    # Create a realistic latent recovery score.
    # This is used only to generate the target variable.
    # ---------------------------------------------------------

    score = (
        1.5 * previous_success_rate
        - 0.30 * attempt_number
        - 0.04 * days_since_failure
        - 0.20 * is_subscription
        + 0.15 * (customer_segment == "loyal")
        + 0.30 * (customer_segment == "high_value")
        + 0.25 * (failure_reason == "bank_timeout")
        + 0.20 * (failure_reason == "network_error")
        - 0.35 * (failure_reason == "insufficient_funds")
        - 0.30 * (failure_reason == "authentication_failed")
        - 0.25 * (failure_reason == "customer_abandoned")
        - 0.01 * invoice_age
    )

    probability = 1 / (1 + np.exp(-score))

    recovered = rng.binomial(1, probability)

    # Recovery action is generated from business logic.
    recovery_action = np.select(
        [
            recovered == 1,
            (attempt_number >= 3) | (days_since_failure >= 7),
            failure_reason == "insufficient_funds",
            failure_reason == "customer_abandoned",
        ],
        [
            "retry_payment",
            "escalate",
            "send_payment_reminder",
            "send_checkout_reminder",
        ],
        default="retry_payment",
    )

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "amount": amount,
            "payment_method": payment_method,
            "event_type": event_type,
            "failure_reason": failure_reason,
            "attempt_number": attempt_number,
            "customer_segment": customer_segment,
            "days_since_failure": days_since_failure,
            "previous_success_rate": previous_success_rate,
            "customer_lifetime_value": customer_lifetime_value,
            "is_subscription": is_subscription,
            "invoice_age": invoice_age,
            "recovered": recovered,
            "recovery_action": recovery_action,
        }
    )

    return df


def main() -> None:
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "recovery_events.csv"

    df = generate_data()

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} transactions")
    print(f"Saved to: {output_path}")
    print()
    print("Recovery rate:")
    print(f"{df['recovered'].mean():.2%}")
    print()
    print("Total revenue at risk:")
    print(f"₹{df['amount'].sum():,.2f}")
    print()
    print("Recovered revenue:")
    print(
        f"₹{df.loc[df['recovered'] == 1, 'amount'].sum():,.2f}"
    )
    print()
    print("Event distribution:")
    print(df["event_type"].value_counts())
    print()
    print("Recovery actions:")
    print(df["recovery_action"].value_counts())


if __name__ == "__main__":
    main()