from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.services.recovery_service import RecoveryService

router = APIRouter()
recovery_service = RecoveryService()

@router.post("/evaluate/{transaction_id}")
def evaluate_transaction(transaction_id: str, db: Session = Depends(get_db)):
    try:
        result = recovery_service.process_transaction(db, transaction_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
