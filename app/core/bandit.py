"""
Contextual Multi-Armed Bandit (CMAB) Engine for RecoverAI.

Uses Thompson Sampling with Bayesian Linear Regression to learn
which recovery action works best for each customer context.

Each arm (recovery action) maintains its own Bayesian linear model.
On each decision, we sample weights from the posterior and pick the
arm with the highest predicted reward for the given context.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.policies import RecoveryAction


# ── Arms ────────────────────────────────────────────────────────

BANDIT_ARMS: list[str] = [a.value for a in RecoveryAction]

# ── Context feature names (for documentation / debugging) ──────

CONTEXT_FEATURES = [
    "amount_norm",           # log-scaled transaction amount
    "recovery_probability",  # from XGBoost
    "attempt_number",
    "days_since_failure",
    "previous_success_rate",
    "clv_norm",              # log-scaled customer lifetime value
    "is_subscription",
    "invoice_age_norm",      # log-scaled invoice age
    "segment_premium",       # 1 if premium
    "segment_enterprise",    # 1 if enterprise
    "fail_insufficient",     # 1 if insufficient_funds
    "fail_timeout",          # 1 if bank_timeout
    "fail_expired",          # 1 if card_expired
    "event_payment_failed",  # 1 if payment_failed
    "event_invoice_overdue", # 1 if invoice_overdue
    "bias",                  # constant 1.0
]

CONTEXT_DIM = len(CONTEXT_FEATURES)  # 16


# ── Context extraction ──────────────────────────────────────────

def extract_context_vector(context) -> np.ndarray:
    """
    Convert a RecoveryContext (or dict-like) into a fixed-size
    numerical feature vector for the bandit.

    Returns a 1-D numpy array of length CONTEXT_DIM.
    """
    amount = getattr(context, "amount", 0.0)
    recovery_prob = getattr(context, "recovery_probability", 0.5)
    attempt = getattr(context, "attempt_number", 1)
    days = getattr(context, "days_since_failure", 0) if hasattr(context, "days_since_failure") else 0
    success_rate = getattr(context, "previous_success_rate", 0.85)
    clv = getattr(context, "customer_lifetime_value", 0.0) if hasattr(context, "customer_lifetime_value") else 0.0
    is_sub = float(getattr(context, "is_subscription", False) if hasattr(context, "is_subscription") else 0)
    inv_age = getattr(context, "invoice_age", 0) if hasattr(context, "invoice_age") else 0
    segment = getattr(context, "customer_segment", "standard")
    failure = getattr(context, "failure_reason", "")
    event = getattr(context, "event_type", "")

    vec = np.array([
        np.log1p(amount),                                # amount_norm
        float(recovery_prob),                            # recovery_probability
        float(attempt),                                  # attempt_number
        float(days),                                     # days_since_failure
        float(success_rate),                             # previous_success_rate
        np.log1p(clv),                                   # clv_norm
        is_sub,                                          # is_subscription
        np.log1p(inv_age),                               # invoice_age_norm
        1.0 if segment == "premium" else 0.0,            # segment_premium
        1.0 if segment == "enterprise" else 0.0,         # segment_enterprise
        1.0 if failure == "insufficient_funds" else 0.0, # fail_insufficient
        1.0 if failure == "bank_timeout" else 0.0,       # fail_timeout
        1.0 if failure == "card_expired" else 0.0,       # fail_expired
        1.0 if event == "payment_failed" else 0.0,       # event_payment_failed
        1.0 if event == "invoice_overdue" else 0.0,      # event_invoice_overdue
        1.0,                                             # bias term
    ], dtype=np.float64)

    return vec


# ── Arm model (Bayesian Linear Regression) ──────────────────────

@dataclass
class ArmModel:
    """
    Bayesian linear regression model for a single arm.

    Maintains:
        B  — precision matrix (d×d), starts as I_d
        mu — mean weight vector (d,), starts as zeros
        f  — B @ mu cache (d,), for efficient incremental updates

    Posterior: θ ~ N(mu, v² B⁻¹)
    """
    dim: int
    B: np.ndarray = field(default=None)
    mu: np.ndarray = field(default=None)
    f: np.ndarray = field(default=None)
    pulls: int = 0
    total_reward: float = 0.0

    def __post_init__(self):
        if self.B is None:
            self.B = np.eye(self.dim, dtype=np.float64)
        if self.mu is None:
            self.mu = np.zeros(self.dim, dtype=np.float64)
        if self.f is None:
            self.f = np.zeros(self.dim, dtype=np.float64)

    def sample_theta(self, v: float, rng: np.random.Generator) -> np.ndarray:
        """Sample θ from the posterior N(mu, v² B⁻¹)."""
        try:
            B_inv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            B_inv = np.eye(self.dim)

        cov = v * v * B_inv
        # Ensure symmetry for sampling
        cov = (cov + cov.T) / 2.0
        # Add small regularization for numerical stability
        cov += 1e-8 * np.eye(self.dim)

        theta = rng.multivariate_normal(self.mu, cov)
        return theta

    def predict(self, x: np.ndarray) -> float:
        """Mean predicted reward for context x."""
        return float(x @ self.mu)

    def update(self, x: np.ndarray, reward: float):
        """
        Bayesian update after observing (x, reward).

        B_new = B_old + x xᵀ
        f_new = f_old + reward * x
        mu_new = B_new⁻¹ @ f_new
        """
        self.B = self.B + np.outer(x, x)
        self.f = self.f + reward * x
        try:
            self.mu = np.linalg.solve(self.B, self.f)
        except np.linalg.LinAlgError:
            pass  # keep old mu on singular matrix
        self.pulls += 1
        self.total_reward += reward

    def to_dict(self) -> dict:
        return {
            "B": self.B.tolist(),
            "mu": self.mu.tolist(),
            "f": self.f.tolist(),
            "pulls": self.pulls,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, d: dict, dim: int) -> "ArmModel":
        model = cls(dim=dim)
        model.B = np.array(d["B"], dtype=np.float64)
        model.mu = np.array(d["mu"], dtype=np.float64)
        model.f = np.array(d["f"], dtype=np.float64)
        model.pulls = d.get("pulls", 0)
        model.total_reward = d.get("total_reward", 0.0)
        return model


# ── Contextual Bandit ────────────────────────────────────────────

@dataclass
class ArmSelection:
    """Result of a bandit arm selection."""
    arm: str
    scores: dict[str, float]  # Thompson-sampled score per arm
    exploration_score: float   # score of the selected arm
    context_vector: list[float]


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit using Thompson Sampling.

    Learns which recovery action maximizes reward for each
    customer context by maintaining per-arm Bayesian linear
    regression models.
    """

    DEFAULT_STATE_PATH = "models/bandit_state.json"

    def __init__(
        self,
        v: float = 0.3,
        state_path: Optional[str] = None,
    ):
        """
        Args:
            v: Exploration parameter. Controls posterior width.
               Lower = more exploitation, higher = more exploration.
               0.3 provides strong exploitation with enough
               exploration to discover better strategies.
            state_path: Path to persist bandit state.
        """
        self.v = v
        self.state_path = state_path or self.DEFAULT_STATE_PATH
        self.dim = CONTEXT_DIM
        self.arms: dict[str, ArmModel] = {
            arm: ArmModel(dim=self.dim) for arm in BANDIT_ARMS
        }
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed=42)
        self._total_rounds = 0

        # Try to load saved state
        self._load_state()

    def select_arm(self, context_vector: np.ndarray) -> ArmSelection:
        """
        Select the best arm for the given context using
        Thompson Sampling.

        Returns an ArmSelection with the chosen arm and all
        Thompson-sampled scores.
        """
        with self._lock:
            scores = {}
            for arm_name, arm_model in self.arms.items():
                theta = arm_model.sample_theta(self.v, self._rng)
                score = float(context_vector @ theta)
                scores[arm_name] = round(score, 6)

            best_arm = max(scores, key=scores.get)
            self._total_rounds += 1

        return ArmSelection(
            arm=best_arm,
            scores=scores,
            exploration_score=scores[best_arm],
            context_vector=context_vector.tolist(),
        )

    def update_reward(self, arm: str, context_vector: np.ndarray, reward: float):
        """
        Update the arm's Bayesian model with an observed reward.

        Args:
            arm: The arm that was pulled.
            context_vector: The context when the arm was pulled.
            reward: The observed reward (0.0 to 1.0).
        """
        if arm not in self.arms:
            return

        with self._lock:
            self.arms[arm].update(context_vector, reward)

        # Persist state after update
        self._save_state()

    def get_stats(self) -> dict:
        """Return summary statistics for all arms."""
        with self._lock:
            stats = {
                "total_rounds": self._total_rounds,
                "exploration_param": self.v,
                "arms": {},
            }
            for arm_name, arm_model in self.arms.items():
                avg_reward = (
                    arm_model.total_reward / arm_model.pulls
                    if arm_model.pulls > 0
                    else 0.0
                )
                stats["arms"][arm_name] = {
                    "pulls": arm_model.pulls,
                    "total_reward": round(arm_model.total_reward, 4),
                    "avg_reward": round(avg_reward, 4),
                    "mean_weights": arm_model.mu.tolist(),
                }

            # Determine top arm
            if any(m.pulls > 0 for m in self.arms.values()):
                top = max(
                    self.arms.items(),
                    key=lambda x: (
                        x[1].total_reward / x[1].pulls
                        if x[1].pulls > 0
                        else -1
                    ),
                )
                stats["top_arm"] = top[0]
                stats["top_arm_avg_reward"] = round(
                    top[1].total_reward / top[1].pulls
                    if top[1].pulls > 0
                    else 0.0,
                    4,
                )
            else:
                stats["top_arm"] = None
                stats["top_arm_avg_reward"] = 0.0

            return stats

    def reset(self):
        """Reset all arm models to their priors."""
        with self._lock:
            self.arms = {
                arm: ArmModel(dim=self.dim) for arm in BANDIT_ARMS
            }
            self._total_rounds = 0
            self._rng = np.random.default_rng(seed=42)

        # Remove saved state
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def _save_state(self):
        """Persist bandit state to JSON."""
        state = {
            "v": self.v,
            "total_rounds": self._total_rounds,
            "arms": {
                name: model.to_dict()
                for name, model in self.arms.items()
            },
        }
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f)
        except (OSError, IOError):
            pass  # Non-critical: state will be rebuilt

    def _load_state(self):
        """Load bandit state from JSON if it exists."""
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)

            self.v = state.get("v", self.v)
            self._total_rounds = state.get("total_rounds", 0)

            for arm_name, arm_data in state.get("arms", {}).items():
                if arm_name in self.arms:
                    self.arms[arm_name] = ArmModel.from_dict(
                        arm_data, self.dim
                    )
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # Start fresh on corrupt state


# ── Module-level singleton ──────────────────────────────────────

bandit = ContextualBandit()
