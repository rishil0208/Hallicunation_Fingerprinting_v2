"""FastAPI application — 7 endpoints per spec Section 5.12."""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.app.fingerprint.cluster import FEATURE_KEYS, build_fingerprint
from backend.app.gate.calibrate import calibrate_per_model, ensure_primary_only, split_records
from backend.app.gate.escalate import score_answer
from backend.app.gate.score import UncalibratedModelError
from backend.app.plugins.features.default import DefaultFeatureExtractor
from backend.app.plugins.judges.gemini_judge import GeminiJudgeError
from backend.app.plugins.judges.mock_judge import MockJudgePlugin
from backend.app.registry import (
    register_feature_extractor,
    register_judge,
)
from backend.app.schemas import Fingerprint

# ── In-memory stores ──
_fingerprints: dict[str, Fingerprint] = {}
_jobs: dict[str, dict] = {}
_eval_results: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Register default plugins on startup."""
    register_feature_extractor(DefaultFeatureExtractor())
    register_judge(MockJudgePlugin())
    yield


app = FastAPI(
    title="Hallucination Fingerprinting Gate",
    description="Three-stage hallucination detection: Features → Symbolic Gate → LLM Judge",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/Response schemas ──

class ScoreRequest(BaseModel):
    answer: str
    model_id: str


class ScoreResponse(BaseModel):
    verdict: str
    gate_score: float
    thresholds: dict
    resolved_by: str
    triggered_patterns: list[dict]
    feature_breakdown: dict[str, float]
    explanation: str
    explanation_source: str
    model_id: str
    fingerprint_version: str


class ModelInfo(BaseModel):
    model_id: str
    fingerprint_version: str
    calibration_dataset_size: int
    last_calibrated_at: str


class CalibrateResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    status: str
    result: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ── POST /api/v1/score ──

@app.post("/api/v1/score", response_model=ScoreResponse)
def score_endpoint(request: ScoreRequest):
    """Score an answer for hallucination risk."""
    if request.model_id not in _fingerprints:
        raise HTTPException(
            status_code=404,
            detail=f"No fingerprint found for model '{request.model_id}'. Calibrate first.",
        )

    fingerprint = _fingerprints[request.model_id]

    try:
        result = score_answer(
            answer=request.answer,
            model_id=request.model_id,
            fingerprint=fingerprint,
        )
    except UncalibratedModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiJudgeError as e:
        # ADR A-C1: judge failure → HTTP 502
        raise HTTPException(status_code=502, detail=f"Judge error: {e}")

    return ScoreResponse(
        verdict=result.verdict,
        gate_score=result.gate_score,
        thresholds={
            "t_low": result.thresholds.t_low,
            "t_high": result.thresholds.t_high,
        },
        resolved_by=result.resolved_by,
        triggered_patterns=[
            {
                "name": p.name,
                "features_involved": p.features_involved,
                "strength": p.strength,
            }
            for p in result.triggered_patterns
        ],
        feature_breakdown=result.feature_breakdown,
        explanation=result.explanation,
        explanation_source=result.explanation_source,
        model_id=result.model_id,
        fingerprint_version=result.fingerprint_version,
    )


# ── GET /api/v1/models ──

@app.get("/api/v1/models", response_model=list[ModelInfo])
def list_models():
    """List all models with calibrated fingerprints."""
    return [
        ModelInfo(
            model_id=fp.model_id,
            fingerprint_version=fp.version,
            calibration_dataset_size=fp.calibration_dataset_size,
            last_calibrated_at=fp.last_calibrated_at,
        )
        for fp in _fingerprints.values()
    ]


# ── GET /api/v1/models/{model_id}/fingerprint ──

@app.get("/api/v1/models/{model_id}/fingerprint")
def get_fingerprint(model_id: str):
    """Get a model's fingerprint summary for visualization."""
    if model_id not in _fingerprints:
        raise HTTPException(status_code=404, detail=f"No fingerprint for '{model_id}'")

    fp = _fingerprints[model_id]
    return {
        "model_id": fp.model_id,
        "version": fp.version,
        "clusters": fp.clusters.model_dump() if fp.clusters else None,
        "w_i": fp.w_i,
        "t_low": fp.t_low,
        "t_high": fp.t_high,
        "normalization": fp.normalization.model_dump(),
        "calibration_dataset_size": fp.calibration_dataset_size,
        "last_calibrated_at": fp.last_calibrated_at,
    }


# ── POST /api/v1/models/{model_id}/calibrate ──

@app.post("/api/v1/models/{model_id}/calibrate", response_model=CalibrateResponse)
def calibrate_model(model_id: str):
    """Trigger background calibration job for a model."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None}

    def _run_calibration():
        try:
            _jobs[job_id]["status"] = "running"

            # Load primary dataset
            from backend.app.plugins.datasets.halueval_qa import HaluEvalQALoader
            data_path = Path("data/raw/qa_data.json")
            if not data_path.exists():
                _jobs[job_id] = {"status": "failed", "result": {"error": "Dataset not found"}}
                return

            loader = HaluEvalQALoader(data_path, default_model_id=model_id)
            records = list(loader.load())
            records = ensure_primary_only(records)

            # Split
            train, val, test = split_records(records)

            # Extract features on training data
            extractor = DefaultFeatureExtractor()
            train_features = [extractor.extract(r.answer) for r in train]
            train_labels = [r.label for r in train]

            # Build fingerprint (clustering on training data)
            fp = build_fingerprint(model_id, train_features, train_labels)

            # Calibrate thresholds on validation data
            from backend.app.gate.score import compute_gate_score

            val_features = [extractor.extract(r.answer) for r in val]
            val_labels = [r.label for r in val]
            val_scores = []
            for vf in val_features:
                eval_result = compute_gate_score(vf, fp)
                val_scores.append(eval_result.gate_score)

            t_low, t_high, w_i = calibrate_per_model(val_scores, val_labels)

            # Update fingerprint with calibrated values
            fp.t_low = t_low
            fp.t_high = t_high
            fp.w_i = w_i
            fp.last_calibrated_at = datetime.now(timezone.utc).isoformat()

            _fingerprints[model_id] = fp
            _jobs[job_id] = {
                "status": "complete",
                "result": {
                    "model_id": model_id,
                    "t_low": t_low,
                    "t_high": t_high,
                    "calibration_dataset_size": fp.calibration_dataset_size,
                },
            }

        except Exception as e:
            _jobs[job_id] = {"status": "failed", "result": {"error": str(e)}}

    thread = Thread(target=_run_calibration, daemon=True)
    thread.start()

    return CalibrateResponse(job_id=job_id, status="pending")


# ── GET /api/v1/jobs/{job_id} ──

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Check background job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = _jobs[job_id]
    return JobStatus(status=job["status"], result=job.get("result"))


# ── GET /api/v1/eval/summary ──

@app.get("/api/v1/eval/summary")
def eval_summary():
    """Return latest evaluation metrics."""
    results_dir = Path("eval/results")
    if not results_dir.exists():
        return {"experiments": [], "message": "No evaluation results yet."}

    experiments = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fh:
                experiments.append(json.load(fh))
        except Exception:
            continue

    return {"experiments": experiments}


# ── GET /api/v1/health ──

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    """Basic liveness check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Helper: register a pre-built fingerprint (for testing) ──

def register_fingerprint(fingerprint: Fingerprint) -> None:
    """Register a fingerprint in the in-memory store."""
    _fingerprints[fingerprint.model_id] = fingerprint
