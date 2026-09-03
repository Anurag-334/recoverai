import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.api import payments, recovery, webhooks
from app.database.database import engine, Base, SessionLocal
from app.services.razorpay_service import RazorpayMockService

# Create database tables
Base.metadata.create_all(bind=engine)

# Seed database on startup if empty
db = SessionLocal()
from app.database.models import Transaction
if db.query(Transaction).count() == 0:
    seeder = RazorpayMockService()
    seeder.seed_database(db)
db.close()

app = FastAPI(title="AI Revenue Recovery")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

app.include_router(payments.router, prefix="/api/payments", tags=["payments"])
app.include_router(recovery.router, prefix="/api/recovery", tags=["recovery"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/audit", response_class=HTMLResponse)
async def audit(request: Request, transaction_id: str):
    return templates.TemplateResponse(request=request, name="audit.html", context={"transaction_id": transaction_id})
