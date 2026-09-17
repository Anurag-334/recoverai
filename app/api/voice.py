import json
import base64
import asyncio
import edge_tts
import requests
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Transaction, AuditLog
from app.agents.voice_agent import VoiceAgent
from app.core.config import settings

router = APIRouter()

voice_agent = VoiceAgent()

def generate_tts_sync(text: str, language: str) -> str:
    """Generate TTS audio synchronously and return base64 encoded mp3."""
    
    # 1. Try ElevenLabs if API key is provided
    if settings.elevenlabs_api_key:
        try:
            # Rachel voice ID: 21m00Tcm4TlvDq8ikWAM, Alice: Xb7hH8MSALEjdAclc2P1
            # We'll use Alice as she sounds very professional for a billing agent
            voice_id = "Xb7hH8MSALEjdAclc2P1" 
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": settings.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2" if language and language.lower() not in ["en", "english"] else "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode("utf-8")
            else:
                print(f"ElevenLabs API Error ({response.status_code}): {response.text}")
                # Fall through to Edge TTS
        except Exception as e:
            print(f"ElevenLabs Exception: {e}")
            # Fall through to Edge TTS
            
    # 2. Fallback to Microsoft Edge Neural TTS (100% Free)
    voice = "en-US-AriaNeural" # Default US English
    if language and language.lower() in ["hi", "hindi", "hinglish"]:
        voice = "hi-IN-SwaraNeural"
    elif language and language.lower() in ["en-in", "indian english", "english"]:
        voice = "en-IN-NeerjaNeural" # Use Indian English for this app's context

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return base64.b64encode(audio_data).decode("utf-8")

    try:
        return asyncio.run(_generate())
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""


@router.post("/initiate/{transaction_id}")
def initiate_call(transaction_id: str, db: Session = Depends(get_db)):
    """Start a voice call session for a transaction."""
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    customer_name = f"Customer {txn.customer_id}"
    greeting_response = voice_agent.generate_greeting(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        failure_reason=txn.failure_reason or "unknown reason",
        customer_name=customer_name,
        language=txn.language_preference or "English"
    )

    # Load existing history or start fresh
    history = []
    if txn.negotiation_history:
        try:
            history = json.loads(txn.negotiation_history)
        except json.JSONDecodeError:
            pass

    # Mark the start of a voice call session
    history.append({"role": "system", "content": "[VOICE_CALL]"})
    history.append({"role": "assistant", "content": greeting_response.spoken_reply})

    txn.negotiation_history = json.dumps(history)

    # Create audit log entry using the actual AuditLog model fields
    audit_log = AuditLog(
        transaction_id=transaction_id,
        recovery_probability=txn.previous_success_rate or 0.5,
        agent_diagnosis="Voice call initiated with customer",
        agent_recommended_action="voice_call",
        agent_reasoning=f"Initiated voice recovery call. Greeting sent in {txn.language_preference or 'English'}.",
        agent_confidence=0.9,
        policy_allowed=True,
        policy_reason="Voice call approved by policy.",
        policy_expected_value=float(txn.amount),
        generated_message=greeting_response.spoken_reply,
        execution_status="SUCCESS: Voice call initiated.",
        final_action="voice_call_initiated"
    )
    db.add(audit_log)
    db.commit()

    greeting_dict = greeting_response.model_dump()
    audio_b64 = generate_tts_sync(greeting_response.spoken_reply, txn.language_preference or "English")
    if audio_b64:
        greeting_dict["audio_base64"] = audio_b64

    return {
        "status": "success",
        "greeting": greeting_dict,
        "transaction": {
            "id": txn.transaction_id,
            "amount": txn.amount,
            "customer_id": txn.customer_id,
            "failure_reason": txn.failure_reason,
            "language_preference": txn.language_preference
        }
    }


@router.post("/turn/{transaction_id}")
def process_turn(transaction_id: str, payload: dict, db: Session = Depends(get_db)):
    """Process a single conversation turn during a voice call."""
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    message = payload.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Load conversation history
    history = []
    if txn.negotiation_history:
        try:
            history = json.loads(txn.negotiation_history)
        except json.JSONDecodeError:
            pass

    turn_response = voice_agent.process_turn(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        customer_message=message,
        conversation_history=history,
        language=txn.language_preference or "English"
    )

    # Process intent-based actions
    if turn_response.intent == "promise_to_pay":
        txn.promise_to_pay_date = datetime.now(timezone.utc) + timedelta(days=1)
    elif turn_response.intent == "request_split":
        txn.split_payment_active = True

    # Append conversation turns
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": turn_response.spoken_reply})

    txn.negotiation_history = json.dumps(history)
    db.commit()

    reply_dict = turn_response.model_dump()
    audio_b64 = generate_tts_sync(turn_response.spoken_reply, txn.language_preference or "English")
    if audio_b64:
        reply_dict["audio_base64"] = audio_b64

    return {
        "status": "success",
        "reply": reply_dict
    }


@router.post("/end/{transaction_id}")
def end_call(transaction_id: str, db: Session = Depends(get_db)):
    """End a voice call session and create a summary audit log."""
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Load conversation history
    history = []
    if txn.negotiation_history:
        try:
            history = json.loads(txn.negotiation_history)
        except json.JSONDecodeError:
            pass

    summary = voice_agent.generate_summary(transaction_id, history)

    # Create comprehensive audit log
    audit_log = AuditLog(
        transaction_id=transaction_id,
        recovery_probability=txn.previous_success_rate or 0.5,
        agent_diagnosis=f"Voice call completed. Final intent: {summary['final_intent']}",
        agent_recommended_action="voice_call",
        agent_reasoning=f"Completed {summary['duration_turns']}-turn voice recovery call. Final intent: {summary['final_intent']}.",
        agent_confidence=0.9,
        policy_allowed=True,
        policy_reason="Voice call session completed.",
        policy_expected_value=float(txn.amount),
        generated_message=json.dumps(summary.get("transcript", [])[-2:]) if summary.get("transcript") else None,
        execution_status=f"SUCCESS: Voice call ended. {summary['duration_turns']} turns. Intent: {summary['final_intent']}.",
        final_action="voice_call"
    )
    db.add(audit_log)
    db.commit()

    return {
        "status": "success",
        "summary": summary
    }


@router.get("/transactions")
def get_voice_transactions(db: Session = Depends(get_db)):
    """Return all transactions for the phone simulator selector."""
    txns = db.query(Transaction).all()
    result = []
    for txn in txns:
        result.append({
            "transaction_id": txn.transaction_id,
            "customer_id": txn.customer_id,
            "amount": float(txn.amount),
            "failure_reason": txn.failure_reason,
            "language_preference": txn.language_preference or "English",
            "recovered": txn.recovered,
            "payment_method": txn.payment_method
        })
    return result
