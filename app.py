"""FastAPI backend for the Nova Guardrail dashboard (index.html).

Verdict    : Vectara HHEM through halludetect.service.Detector, with the threshold tuned on the RAGTruth TRAIN split in
             Phase 1 (results/phase1_summary.json). This is the evaluated detector.
Experiment : the claim-level verifier, shown next to it as "experimental". It is only available once
             `python -m halludetect phase3` has written results/phase3_summary.json, and it uses the aggregation and
             threshold chosen there. If calibration (Phase 4) was run, its probability and abstention flag are added.
Nothing here is hard-coded: if a model or a results file is missing the API says so instead of inventing numbers.

Run:  python -m halludetect serve      (or: uvicorn app:app --port 8000)
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

RESULTS = ROOT / "results"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once. A failure is recorded and reported by /health and /api/verify, never hidden."""
    app.state.errors = []
    try:
        from halludetect.service import Detector

        app.state.detector = Detector(use_nli=True, use_hhem=True)
        app.state.errors += [f"{name}: {msg}" for name, msg in app.state.detector.load_errors.items()]
    except Exception as err:  # noqa: BLE001
        app.state.errors.append(f"detector: {type(err).__name__}: {err}")

    summary = _read_json(RESULTS / "phase3_summary.json")
    if app.state.detector is not None and summary and "best_aggregation" in summary and "best_threshold" in summary:
        try:
            from halludetect.verifiers.claim_level import ClaimLevelVerifier

            det = app.state.detector
            app.state.claim_verifier = ClaimLevelVerifier(nli=det.nli, device=det.device)
            app.state.claim_summary = summary
        except Exception as err:  # noqa: BLE001
            app.state.errors.append(f"claim_level: {type(err).__name__}: {err}")
    calibrator_path = RESULTS / "phase4_calibrator.json"
    if calibrator_path.exists():
        try:
            from halludetect.calibration import LogisticCalibrator

            app.state.calibrator = LogisticCalibrator.load(calibrator_path)
            app.state.abstain_margin = (_read_json(RESULTS / "phase4_summary.json") or {}).get("abstain_margin")
        except Exception as err:  # noqa: BLE001
            app.state.errors.append(f"calibrator: {type(err).__name__}: {err}")
    yield


app = FastAPI(title="Nova Guardrail API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# defaults, so the routes also work when the lifespan has not run (for example in tests)
app.state.detector = None
app.state.claim_verifier = None
app.state.claim_summary = None
app.state.calibrator = None
app.state.abstain_margin = None
app.state.errors = []


class VerifyRequest(BaseModel):
    context: str
    answer: str
    task: str | None = None
    experimental: bool = True  # False skips the (slower) claim-level check


def _experimental(context: str, answer: str, task: str | None) -> dict:
    base = {"engine": "Claim-level verifier (experimental)"}
    verifier, summary = app.state.claim_verifier, app.state.claim_summary
    if verifier is None or not summary:
        return {**base, "available": False,
                "reason": "results/phase3_summary.json not found. Run `python -m halludetect phase3` first."}
    from halludetect.verifiers.claim_level import hallucination_score

    try:
        result = verifier.verify(context, answer, task)
    except Exception as err:  # noqa: BLE001
        return {**base, "available": False, "reason": f"{type(err).__name__}: {err}"}
    aggregation, tau = summary["best_aggregation"], summary["best_threshold"]
    score = hallucination_score(result["aggs"], aggregation)  # higher = more likely hallucinated
    claim_tau = summary.get("claim_threshold")
    out = {**base, "available": True, "aggregation": aggregation, "score": score, "threshold_used": tau,
           "is_hallucinated": bool(score >= tau),
           "claims": [{"text": c["text"], "support": c["p_entail"], "contradiction": c["p_contra"],
                       "flagged": None if claim_tau is None else bool(1.0 - c["p_entail"] >= claim_tau),
                       "evidence": c.get("evidence", "")[:300]} for c in result["claims"]]}
    calibrator = app.state.calibrator
    if calibrator is not None:
        features = [hallucination_score(result["aggs"], name[2:]) for name in calibrator.feature_names]
        p = float(calibrator.predict_proba(np.array([features]))[0])
        margin = app.state.abstain_margin
        out["probability_hallucinated"] = p
        out["uncertain"] = bool(margin is not None and abs(p - 0.5) < margin)
    return out


@app.get("/health")
def health():
    detector = app.state.detector
    return {"status": "ok" if detector is not None else "unavailable",
            "models_loaded": detector is not None,
            "engines": detector.engines if detector is not None else [],
            "claim_level_ready": app.state.claim_verifier is not None and bool(app.state.claim_summary),
            "calibrated": app.state.calibrator is not None,
            "errors": list(app.state.errors)}


@app.post("/api/verify")
def verify(req: VerifyRequest):
    detector = app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Models are not loaded: " + ("; ".join(app.state.errors) or "still starting up"))
    if not req.context.strip() or not req.answer.strip():
        raise HTTPException(status_code=400, detail="Both context and answer are required.")
    start = time.perf_counter()
    try:
        result = detector.verify(req.context, req.answer, primary="hhem")
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Verification failed: {type(err).__name__}: {err}")
    model = result["models"][result["primary_engine"]]
    return {
        "status": "ok",
        "primary_verdict": {"engine": model["engine"], "model_name": model.get("model_name"), "score": model["score"],
                            "threshold_used": model["tau"], "is_hallucinated": bool(model["flagged"]),
                            "threshold_source": detector.tau_source},
        "engines_agree": result["engines_agree"],
        "models": result["models"],
        "warnings": result["warnings"],
        "experimental": _experimental(req.context, req.answer, req.task) if req.experimental else {
            "engine": "Claim-level verifier (experimental)", "available": False,
            "reason": "Switched off for speed. Tick the box in the sidebar to run it."},
        "latency_ms": (time.perf_counter() - start) * 1000.0,
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")
