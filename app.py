from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from halludetect.verifiers.claim_level import ClaimLevelVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Hallucination Detector API",
    description="Backend API serving the ClaimLevelVerifier for real-time RAG response evaluation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global verifier instance to keep the model loaded in memory
verifier = None

@app.on_event("startup")
def load_verifier():
    global verifier
    logger.info("Initializing ClaimLevelVerifier...")
    try:
        verifier = ClaimLevelVerifier()
        logger.info("ClaimLevelVerifier loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load verifier: {e}")
        raise e

class VerifyRequest(BaseModel):
    context: str = Field(..., description="The reference source text retrieved for RAG.")
    answer: str = Field(..., description="The generated model answer to verify.")
    task: str = Field("qa", description="Evaluation task type.")

@app.post("/api/verify")
def verify_response(request: VerifyRequest):
    if not verifier:
        raise HTTPException(status_code=500, detail="Verifier model is not initialized.")
    try:
        result = verifier.verify(request.context, request.answer, task=request.task)
        score = float(result.get("score", 0.0))
        
        # Using your optimized threshold from Phase 4 evaluation
        is_hallucinated = score < 0.991
        
        return {
            "score": score,
            "is_hallucinated": is_hallucinated,
            "threshold_used": 0.991,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": verifier is not None}
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")