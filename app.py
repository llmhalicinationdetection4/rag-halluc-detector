import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_threshold(filepath, engine, default_val):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            # Assuming phase1 structure based on Claude's notes
            if engine == "hhem" and "thresholds" in data:
                return data["thresholds"]["hhem"]["tuned_tau"]
            return data.get("best_threshold", default_val)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.error(f"Missing threshold file/key for {engine}. Failsafe prevented.")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.hhem_detector = None
    app.state.claim_verifier = None
    try:
        from halludetect.service import Detector
        from halludetect.verifiers.claim_level import ClaimLevelVerifier
        app.state.hhem_detector = Detector()
        app.state.claim_verifier = ClaimLevelVerifier()
        app.state.hhem_tau = load_threshold("results/phase1_summary.json", "hhem", None)
        app.state.claim_tau = load_threshold("results/phase3_summary.json", "claim", None)
        if app.state.hhem_tau is None or app.state.claim_tau is None:
            raise ValueError("Threshold files missing. Will not start with invented defaults.")
        logger.info("Models and thresholds loaded successfully.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
    yield

app = FastAPI(title="RAG Hallucination Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    context: str = Field(..., description="The reference source text.")
    answer: str = Field(..., description="The generated model answer.")

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    loaded = app.state.hhem_detector is not None
    return {"status": "active" if loaded else "unavailable", "models_loaded": loaded}

@app.post("/api/verify")
def verify_response(request: VerifyRequest):
    if app.state.hhem_detector is None or app.state.claim_verifier is None:
        return JSONResponse(status_code=503, content={"detail": "Models are currently unavailable."})

    try:
        # Call verified HHEM pipeline
        hhem_res = app.state.hhem_detector.verify(request.context, request.answer)
        hhem_score = hhem_res["models"]["hhem"]["score"]
        
        # Call claim-level verifier
        claim_res = app.state.claim_verifier.verify(request.context, request.answer)
        p_contras = [c.get("p_contra", 0.0) for c in claim_res.get("claims", [])]
        claim_score = max(p_contras) if p_contras else 0.0

        return {
            "status": "success",
            "primary_verdict": {
                "engine": "HHEM",
                "score": hhem_score,
                "threshold_used": app.state.hhem_tau,
                "is_hallucinated": hhem_score < app.state.hhem_tau
            },
            "experimental": {
                "engine": "ClaimLevelVerifier",
                "score": claim_score,
                "threshold_used": app.state.claim_tau,
                "is_hallucinated": claim_score >= app.state.claim_tau
            }
        }
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))