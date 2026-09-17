"""
Unit tests for the Contextual Multi-Armed Bandit engine.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from app.core.bandit import (
    BANDIT_ARMS,
    CONTEXT_DIM,
    ArmModel,
    ArmSelection,
    ContextualBandit,
    extract_context_vector,
)
from app.core.policies import RecoveryAction


# ── Helper: build a mock context object ─────────────────────────

class MockContext:
    """Mimics RecoveryContext for testing."""

    def __init__(self, **kwargs):
        defaults = {
            "amount": 5000.0,
            "recovery_probability": 0.75,
            "attempt_number": 1,
            "days_since_failure": 2,
            "previous_success_rate": 0.85,
            "customer_lifetime_value": 15000.0,
            "is_subscription": False,
            "invoice_age": 5,
            "customer_segment": "premium",
            "failure_reason": "bank_timeout",
            "event_type": "payment_failed",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ── Tests: Context vector extraction ────────────────────────────


def test_context_vector_dimension():
    """extract_context_vector should return exactly CONTEXT_DIM floats."""
    ctx = MockContext()
    vec = extract_context_vector(ctx)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (CONTEXT_DIM,)
    assert vec.dtype == np.float64


def test_context_vector_bias_term():
    """Last element should always be 1.0 (bias)."""
    ctx = MockContext()
    vec = extract_context_vector(ctx)
    assert vec[-1] == 1.0


def test_context_vector_segment_encoding():
    """One-hot encoding of customer segments."""
    premium = extract_context_vector(MockContext(customer_segment="premium"))
    enterprise = extract_context_vector(MockContext(customer_segment="enterprise"))
    standard = extract_context_vector(MockContext(customer_segment="standard"))

    # premium: index 8 = 1, index 9 = 0
    assert premium[8] == 1.0
    assert premium[9] == 0.0

    # enterprise: index 8 = 0, index 9 = 1
    assert enterprise[8] == 0.0
    assert enterprise[9] == 1.0

    # standard: both 0
    assert standard[8] == 0.0
    assert standard[9] == 0.0


def test_context_vector_failure_encoding():
    """One-hot encoding of failure reasons."""
    insufficient = extract_context_vector(
        MockContext(failure_reason="insufficient_funds")
    )
    timeout = extract_context_vector(
        MockContext(failure_reason="bank_timeout")
    )
    expired = extract_context_vector(
        MockContext(failure_reason="card_expired")
    )
    other = extract_context_vector(
        MockContext(failure_reason="unknown_error")
    )

    # insufficient_funds: index 10 = 1
    assert insufficient[10] == 1.0
    assert insufficient[11] == 0.0

    # bank_timeout: index 11 = 1
    assert timeout[10] == 0.0
    assert timeout[11] == 1.0

    # card_expired: index 12 = 1
    assert expired[12] == 1.0

    # unknown: all failure flags 0
    assert other[10] == 0.0
    assert other[11] == 0.0
    assert other[12] == 0.0


def test_context_vector_log_scaling():
    """Amount and CLV should be log-scaled."""
    ctx = MockContext(amount=10000.0, customer_lifetime_value=50000.0)
    vec = extract_context_vector(ctx)

    assert vec[0] == pytest.approx(np.log1p(10000.0))
    assert vec[5] == pytest.approx(np.log1p(50000.0))


# ── Tests: ArmModel ─────────────────────────────────────────────


def test_arm_model_initial_state():
    """New ArmModel should start with identity precision and zero weights."""
    model = ArmModel(dim=4)

    np.testing.assert_array_equal(model.B, np.eye(4))
    np.testing.assert_array_equal(model.mu, np.zeros(4))
    assert model.pulls == 0
    assert model.total_reward == 0.0


def test_arm_model_update():
    """After an update, pulls and total_reward should increase."""
    model = ArmModel(dim=3)
    x = np.array([1.0, 0.5, 0.3])

    model.update(x, reward=1.0)

    assert model.pulls == 1
    assert model.total_reward == 1.0
    # B should no longer be identity
    assert not np.allclose(model.B, np.eye(3))


def test_arm_model_serialization():
    """to_dict / from_dict should round-trip correctly."""
    model = ArmModel(dim=3)
    x = np.array([1.0, 0.5, 0.2])
    model.update(x, reward=0.7)

    data = model.to_dict()
    restored = ArmModel.from_dict(data, dim=3)

    np.testing.assert_array_almost_equal(model.B, restored.B)
    np.testing.assert_array_almost_equal(model.mu, restored.mu)
    np.testing.assert_array_almost_equal(model.f, restored.f)
    assert model.pulls == restored.pulls
    assert model.total_reward == pytest.approx(restored.total_reward)


def test_arm_model_predict():
    """Predict should return dot product of x and mu."""
    model = ArmModel(dim=3)
    model.mu = np.array([0.5, 1.0, -0.3])
    x = np.array([2.0, 1.0, 1.0])

    # 0.5*2 + 1.0*1 + (-0.3)*1 = 1.7
    assert model.predict(x) == pytest.approx(1.7)


# ── Tests: ContextualBandit ──────────────────────────────────────


def test_select_arm_returns_valid_action():
    """Selected arm should be a valid RecoveryAction value."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/path.json")
    ctx = MockContext()
    vec = extract_context_vector(ctx)

    result = b.select_arm(vec)

    assert isinstance(result, ArmSelection)
    assert result.arm in BANDIT_ARMS
    assert result.arm in [a.value for a in RecoveryAction]
    assert len(result.scores) == len(BANDIT_ARMS)
    assert len(result.context_vector) == CONTEXT_DIM


def test_select_arm_scores_all_arms():
    """Every arm should have a score in the selection result."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/path.json")
    vec = extract_context_vector(MockContext())

    result = b.select_arm(vec)

    for arm in BANDIT_ARMS:
        assert arm in result.scores


def test_reward_update_changes_stats():
    """After updating with a reward, the arm stats should change."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/path.json")
    vec = extract_context_vector(MockContext())

    selection = b.select_arm(vec)
    arm = selection.arm

    stats_before = b.get_stats()
    pulls_before = stats_before["arms"][arm]["pulls"]

    b.update_reward(arm, vec, reward=1.0)

    stats_after = b.get_stats()
    assert stats_after["arms"][arm]["pulls"] == pulls_before + 1
    assert stats_after["arms"][arm]["total_reward"] > stats_before["arms"][arm]["total_reward"]


def test_get_stats_structure():
    """Stats should have the expected keys."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/path.json")

    stats = b.get_stats()

    assert "total_rounds" in stats
    assert "exploration_param" in stats
    assert "arms" in stats
    assert "top_arm" in stats
    assert "top_arm_avg_reward" in stats
    assert len(stats["arms"]) == len(BANDIT_ARMS)


def test_reset_clears_state():
    """Reset should restore the bandit to initial state."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/path.json")
    vec = extract_context_vector(MockContext())

    # Do some pulls and updates
    for _ in range(5):
        sel = b.select_arm(vec)
        b.update_reward(sel.arm, vec, reward=1.0)

    assert b.get_stats()["total_rounds"] >= 5

    b.reset()

    stats = b.get_stats()
    assert stats["total_rounds"] == 0
    for arm_stats in stats["arms"].values():
        assert arm_stats["pulls"] == 0
        assert arm_stats["total_reward"] == 0.0


def test_persistence_round_trip(tmp_path):
    """Save and load should preserve bandit state."""
    state_path = str(tmp_path / "test_bandit.json")

    b1 = ContextualBandit(v=0.3, state_path=state_path)
    vec = extract_context_vector(MockContext())

    # Train a bit
    for _ in range(3):
        sel = b1.select_arm(vec)
        b1.update_reward(sel.arm, vec, reward=0.8)
    b1._save_state()

    stats1 = b1.get_stats()

    # Load into a new bandit
    b2 = ContextualBandit(v=0.3, state_path=state_path)

    stats2 = b2.get_stats()

    assert stats2["total_rounds"] == stats1["total_rounds"]
    for arm in BANDIT_ARMS:
        assert stats2["arms"][arm]["pulls"] == stats1["arms"][arm]["pulls"]
        assert stats2["arms"][arm]["total_reward"] == pytest.approx(
            stats1["arms"][arm]["total_reward"]
        )


def test_deterministic_with_seed():
    """With same state and seed, two bandits should make the same choice."""
    b1 = ContextualBandit(v=0.3, state_path="/nonexistent/1.json")
    b2 = ContextualBandit(v=0.3, state_path="/nonexistent/2.json")

    vec = extract_context_vector(MockContext())

    # Both start fresh with seed=42
    sel1 = b1.select_arm(vec)
    sel2 = b2.select_arm(vec)

    assert sel1.arm == sel2.arm
    assert sel1.scores == sel2.scores


def test_learning_convergence():
    """
    After many rounds of rewarding one arm exclusively,
    the bandit should strongly prefer that arm.
    """
    b = ContextualBandit(v=0.1, state_path="/nonexistent/conv.json")
    vec = extract_context_vector(MockContext())

    target_arm = "retry_payment"

    # Train: always reward retry_payment, never reward others
    for _ in range(50):
        b.update_reward(target_arm, vec, reward=1.0)
        for other in BANDIT_ARMS:
            if other != target_arm:
                b.update_reward(other, vec, reward=0.0)

    # Now check that retry_payment is chosen most of the time
    choices = []
    for _ in range(20):
        sel = b.select_arm(vec)
        choices.append(sel.arm)

    retry_count = choices.count(target_arm)
    # Should strongly prefer retry_payment (at least 80% of the time)
    assert retry_count >= 16, (
        f"Expected retry_payment to dominate, but got {retry_count}/20"
    )


def test_top_arm_identification():
    """get_stats should correctly identify the top-performing arm."""
    b = ContextualBandit(v=0.3, state_path="/nonexistent/top.json")
    vec = extract_context_vector(MockContext())

    # Give send_payment_reminder the best average
    b.update_reward("send_payment_reminder", vec, reward=1.0)
    b.update_reward("send_payment_reminder", vec, reward=0.9)
    b.update_reward("retry_payment", vec, reward=0.2)

    stats = b.get_stats()
    assert stats["top_arm"] == "send_payment_reminder"
    assert stats["top_arm_avg_reward"] > 0.9
