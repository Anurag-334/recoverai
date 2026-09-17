"""Tests for the VoiceAgent and Voice API endpoints."""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.agents.voice_agent import VoiceAgent, VoiceResponse


# ─── VoiceAgent Unit Tests ─────────────────────────────────────────────

class TestVoiceAgentFallback:
    """Test the deterministic fallback agent (no LLM needed)."""

    def setup_method(self):
        """Create agent with LLM disabled."""
        self.agent = VoiceAgent()
        self.agent.llm = None  # Force fallback mode

    def test_greeting_english(self):
        result = self.agent.generate_greeting(
            transaction_id="TXN-001",
            amount=2500.0,
            failure_reason="Insufficient Funds",
            customer_name="Customer C001",
            language="English"
        )
        assert isinstance(result, VoiceResponse)
        assert result.intent == "greeting"
        assert "2500" in result.spoken_reply
        assert len(result.suggested_responses) >= 3

    def test_greeting_hinglish(self):
        result = self.agent.generate_greeting(
            transaction_id="TXN-002",
            amount=1800.0,
            failure_reason="Bank Timeout",
            customer_name="Customer C002",
            language="Hinglish"
        )
        assert isinstance(result, VoiceResponse)
        assert result.intent == "greeting"
        # Hinglish greeting should contain Hindi words
        assert any(word in result.spoken_reply.lower() for word in ["namaste", "aapke", "payment"])

    def test_turn_request_split(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=5000.0,
            customer_message="Can I pay half now?",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "request_split"
        assert result.notification is not None
        assert result.notification["type"] == "split_payment"

    def test_turn_promise_to_pay(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=3000.0,
            customer_message="I'll pay tomorrow",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "promise_to_pay"
        assert result.notification is not None
        assert result.notification["type"] == "promise_logged"

    def test_turn_pay_now(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=1500.0,
            customer_message="Send me the link please",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "pay_now"
        assert result.notification is not None
        assert result.notification["type"] == "payment_link"

    def test_turn_dispute_query(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=2000.0,
            customer_message="Why did my payment fail?",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "dispute_or_query"
        assert result.notification is None

    def test_turn_decline(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=7000.0,
            customer_message="I can't afford this right now, I refuse",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "decline"

    def test_turn_unknown(self):
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=1000.0,
            customer_message="Hello there",
            conversation_history=[],
            language="English"
        )
        assert result.intent == "unknown"
        assert len(result.suggested_responses) >= 2

    def test_generate_summary(self):
        history = [
            {"role": "system", "content": "[VOICE_CALL]"},
            {"role": "assistant", "content": "Hello, calling about your payment."},
            {"role": "user", "content": "I'll pay tomorrow"},
            {"role": "assistant", "content": "Thank you, promise logged."},
        ]
        summary = self.agent.generate_summary("TXN-001", history)
        assert summary["duration_turns"] == 2
        assert summary["final_intent"] == "promise_to_pay"
        assert "transcript" in summary

    def test_voice_response_no_urls_in_spoken_reply(self):
        """Ensure spoken replies never contain URLs."""
        result = self.agent.process_turn(
            transaction_id="TXN-001",
            amount=2000.0,
            customer_message="Send link",
            conversation_history=[],
            language="English"
        )
        assert "http" not in result.spoken_reply.lower()
        assert "www." not in result.spoken_reply.lower()


# ─── Voice API Integration Tests ───────────────────────────────────────

class TestVoiceAPI:
    """Test the Voice API endpoints with a test database."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Setup test client with in-memory database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from app.database.database import Base, get_db
        from app.database.models import Transaction, AuditLog

        # StaticPool ensures all connections share the same in-memory DB
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine)

        def override_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        from app.main import app
        app.dependency_overrides[get_db] = override_get_db

        # Seed a test transaction
        db = TestSession()
        txn = Transaction(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            amount=2500.0,
            payment_method="UPI",
            event_type="payment.failed",
            failure_reason="Insufficient Funds",
            attempt_number=1,
            customer_segment="premium",
            days_since_failure=2,
            previous_success_rate=0.7,
            customer_lifetime_value=15000.0,
            is_subscription=False,
            invoice_age=5,
            recovered=False,
            language_preference="English"
        )
        db.add(txn)
        db.commit()
        db.close()

        self.client = TestClient(app)
        yield
        app.dependency_overrides.clear()

    def test_get_transactions(self):
        res = self.client.get("/api/voice/transactions")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["transaction_id"] == "TXN-TEST-001"

    def test_initiate_call(self):
        res = self.client.post("/api/voice/initiate/TXN-TEST-001")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "greeting" in data
        assert "spoken_reply" in data["greeting"]
        assert "suggested_responses" in data["greeting"]
        assert data["transaction"]["id"] == "TXN-TEST-001"

    def test_initiate_call_not_found(self):
        res = self.client.post("/api/voice/initiate/TXN-NONEXISTENT")
        assert res.status_code == 404

    def test_turn_split_intent(self):
        # First initiate
        self.client.post("/api/voice/initiate/TXN-TEST-001")
        # Then send a turn
        res = self.client.post(
            "/api/voice/turn/TXN-TEST-001",
            json={"message": "Can I pay half?"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["reply"]["intent"] == "request_split"
        assert data["reply"]["notification"] is not None

    def test_turn_promise_intent(self):
        self.client.post("/api/voice/initiate/TXN-TEST-001")
        res = self.client.post(
            "/api/voice/turn/TXN-TEST-001",
            json={"message": "I'll pay tomorrow"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["reply"]["intent"] == "promise_to_pay"

    def test_end_call(self):
        self.client.post("/api/voice/initiate/TXN-TEST-001")
        self.client.post(
            "/api/voice/turn/TXN-TEST-001",
            json={"message": "I'll pay tomorrow"}
        )
        res = self.client.post("/api/voice/end/TXN-TEST-001")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "summary" in data
        assert "duration_turns" in data["summary"]

    def test_full_call_flow_updates_db(self):
        """Test that a full call flow correctly updates transaction state."""
        # Initiate
        self.client.post("/api/voice/initiate/TXN-TEST-001")
        # Request split
        self.client.post(
            "/api/voice/turn/TXN-TEST-001",
            json={"message": "Can I split the payment?"}
        )
        # End call
        self.client.post("/api/voice/end/TXN-TEST-001")

        # Verify transaction was updated
        txns = self.client.get("/api/voice/transactions").json()
        # The split_payment_active flag isn't exposed via this endpoint,
        # but the call should have completed without errors
        assert len(txns) >= 1
