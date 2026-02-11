from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import os
import numpy as np
import logging
from src.utils.config import settings
from src.rag.engine import get_rag_response

app = FastAPI(title="Support Triage Copilot")

models = {}


@app.on_event("startup")
def load_models():
    try:
        models["cat"] = joblib.load(
            os.path.join(settings.ARTIFACTS_DIR, "category_model.joblib")
        )
        models["pri"] = joblib.load(
            os.path.join(settings.ARTIFACTS_DIR, "priority_model.joblib")
        )
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(
            f"Warning: Could not load models. Ensure artifacts exist. Error: {e}"
        )


class TicketRequest(BaseModel):
    subject: str
    body: str


class TriageResponse(BaseModel):
    category: str
    priority: str
    confidence: Dict[str, float]
    needs_human_review: bool = False


class AnswerResponse(BaseModel):
    triage: TriageResponse
    draft_reply: str
    internal_next_steps: List[str]
    citations: List[Dict[str, str]]


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": len(models) == 2}


@app.post("/triage", response_model=TriageResponse)
def predict_triage(ticket: TicketRequest):
    if "cat" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    text = f"{ticket.subject} {ticket.body}"

    # Predict
    cat = models["cat"].predict([text])[0]
    pri = models["pri"].predict([text])[0]

    # Get Confidence (Max probability)
    cat_prob = np.max(models["cat"].predict_proba([text]))
    pri_prob = np.max(models["pri"].predict_proba([text]))

    return {
        "category": cat,
        "priority": pri,
        "confidence": {"category": float(cat_prob), "priority": float(pri_prob)},
    }


@app.post("/answer", response_model=AnswerResponse)
def generate_answer(ticket: TicketRequest):
    triage_result = predict_triage(ticket)

    triage_data_for_engine = {
        "category": triage_result["category"],
        "priority": triage_result["priority"],
        "category_confidence": triage_result["confidence"]["category"],
    }

    rag_result = get_rag_response(
        query=f"{ticket.subject}\n{ticket.body}", triage_data=triage_data_for_engine
    )

    return {
        "triage": triage_result,
        "draft_reply": rag_result["draft_reply"],
        "internal_next_steps": rag_result["internal_next_steps"],
        "citations": rag_result["citations"],
    }
