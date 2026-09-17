"""
API routes for the Contextual Multi-Armed Bandit system.

Provides endpoints for:
- Viewing bandit learning statistics
- Recording outcome rewards
- Resetting bandit state (for demos)
"""

import json
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.bandit import bandit
from app.database.database import get_db
from app.database.models import BanditEvent

router = APIRouter()


@router.get("/stats")
def get_bandit_stats():
    """Return current bandit arm statistics and learning progress."""
    return bandit.get_stats()


@router.post("/reward/{transaction_id}")
def record_reward(
    transaction_id: str,
    payload: dict,
    db: Session = Depends(get_db),
):
    """
    Record an outcome reward for a bandit event.

    Payload:
        {"reward": 1.0}  — full recovery
        {"reward": 0.5}  — partial (split payment)
        {"reward": 0.3}  — partial (promise to pay)
        {"reward": 0.0}  — no recovery / churned
    """
    reward = payload.get("reward")
    if reward is None or not (0.0 <= reward <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="reward must be a float between 0.0 and 1.0",
        )

    # Find the most recent unrewarded bandit event for this transaction
    event = (
        db.query(BanditEvent)
        .filter(
            BanditEvent.transaction_id == transaction_id,
            BanditEvent.reward.is_(None),
        )
        .order_by(BanditEvent.timestamp.desc())
        .first()
    )

    if not event:
        raise HTTPException(
            status_code=404,
            detail=f"No pending bandit event found for {transaction_id}",
        )

    # Update the event record
    event.reward = reward
    event.reward_observed_at = datetime.now(timezone.utc)
    db.commit()

    # Update the bandit model
    context_vector = np.array(json.loads(event.context_vector), dtype=np.float64)
    bandit.update_reward(event.arm_selected, context_vector, reward)

    return {
        "status": "success",
        "transaction_id": transaction_id,
        "arm": event.arm_selected,
        "reward": reward,
        "total_rounds": bandit.get_stats()["total_rounds"],
    }


@router.post("/reset")
def reset_bandit(db: Session = Depends(get_db)):
    """
    Reset the bandit to its initial state.
    Clears all learned weights and bandit event records.
    Useful for demos and testing.
    """
    bandit.reset()

    # Clear bandit events from DB
    db.query(BanditEvent).delete()
    db.commit()

    return {
        "status": "success",
        "message": "Bandit state reset to initial priors.",
    }


@router.get("/events/{transaction_id}")
def get_bandit_events(
    transaction_id: str,
    db: Session = Depends(get_db),
):
    """Get all bandit events for a specific transaction."""
    events = (
        db.query(BanditEvent)
        .filter(BanditEvent.transaction_id == transaction_id)
        .order_by(BanditEvent.timestamp.desc())
        .all()
    )

    return [
        {
            "id": e.id,
            "transaction_id": e.transaction_id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "arm_selected": e.arm_selected,
            "arm_scores": json.loads(e.arm_scores) if e.arm_scores else {},
            "exploration_score": e.exploration_score,
            "reward": e.reward,
            "reward_observed_at": (
                e.reward_observed_at.isoformat()
                if e.reward_observed_at
                else None
            ),
        }
        for e in events
    ]
