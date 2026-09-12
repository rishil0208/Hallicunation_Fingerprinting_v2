"""FastAPI application — 7 endpoints per spec Section 5.12."""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.fingerprint.cluster import FEATURE_KEYS, build_fingerprint
from backend.app.gate.calibrate import calibrate_per_model, ensure_primary_only, split_records
from backend.app.gate.escalate import score_answer
from backend.app.gate.score import UncalibratedModelError
from backend.app.plugins.features.default import DefaultFeatureExtractor
from backend.app.plugins.judges.gemini_judge import GeminiJudgeError
from backend.app.plugins.judges.mock_judge import MockJudgePlugin
from backend.app.plugins.judges.qwen_judge import QwenJudgeError, QwenJudgePlugin
from backend.app.registry import (
    register_feature_extractor,
    register_judge,
)
from backend.app.schemas import Fingerprint

# ── In-memory stores ──
_fingerprints: dict[str, Fingerprint] = {}
_jobs: dict[str, dict] = {}
_eval_results: dict[str, Any] = {}
_calibration_lock = Lock()  # A-2 fix: serialize calibration writes
_active_calibrations: set[str] = set()  # A-2 fix: one job per model
FP_STORAGE_DIR = Path("data/processed/fingerprints")


def load_stored_fingerprints() -> None:
    """Load pre-calibrated fingerprints from disk into memory."""
    if not FP_STORAGE_DIR.exists():
        FP_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for fp_file in sorted(FP_STORAGE_DIR.glob("*.json")):
        try:
            with open(fp_file, "r") as f:
                data = json.load(f)
                fp = Fingerprint(**data)
                _fingerprints[fp.model_id] = fp
        except Exception:
            pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Register default plugins on startup."""
    import os
    from dotenv import load_dotenv
    load_dotenv(".env")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    extractor = DefaultFeatureExtractor()
    register_feature_extractor(extractor)
    register_judge(MockJudgePlugin())
    register_judge(QwenJudgePlugin())
    
    if os.environ.get("GEMINI_API_KEY"):
        from backend.app.plugins.judges.gemini_judge import GeminiJudgePlugin
        register_judge(GeminiJudgePlugin())

    load_stored_fingerprints()

    # Pre-warm extractor so live inference requests evaluate in milliseconds
    try:
        extractor.extract("Initial warmup sentence for model cache.")
    except Exception:
        pass

    yield


app = FastAPI(
    title="Hallucination Fingerprinting Gate",
    description="Three-stage hallucination detection: Features → Symbolic Gate → LLM Judge",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/Response schemas ──

# A-1 fix: max_length prevents DoS via spaCy tokenization of huge payloads
MAX_ANSWER_LENGTH = 50_000

class ScoreRequest(BaseModel):
    answer: str = Field(..., max_length=MAX_ANSWER_LENGTH)
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

    import os
    judge_override = os.environ.get("JUDGE_NAME")
    if judge_override:
        judge_to_use = judge_override
    elif os.environ.get("USE_QWEN_JUDGE", "1").lower() in ("1", "true", "yes"):
        judge_to_use = "qwen"
    elif os.environ.get("GEMINI_API_KEY"):
        judge_to_use = "gemini"
    else:
        judge_to_use = "mock"

    try:
        result = score_answer(
            answer=request.answer,
            model_id=request.model_id,
            fingerprint=fingerprint,
            judge_name=judge_to_use,
        )
    except UncalibratedModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (GeminiJudgeError, QwenJudgeError) as e:
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
    # A-2 fix: reject duplicate concurrent calibrations for same model
    with _calibration_lock:
        if model_id in _active_calibrations:
            raise HTTPException(
                status_code=409,
                detail=f"Calibration already in progress for '{model_id}'",
            )
        _active_calibrations.add(model_id)

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

            # CR-3 fix: cap records for practical calibration time
            max_samples = 2000
            if len(records) > max_samples:
                import random
                rng = random.Random(42)
                records = rng.sample(records, max_samples)

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
            from backend.app.gate.rules import compute_fact_activations, compute_rule_activations
            from backend.app.fingerprint.cluster import normalize_features

            val_features = [extractor.extract(r.answer) for r in val]
            val_labels = [r.label for r in val]
            val_scores = []
            val_rule_activations = []
            for vf in val_features:
                eval_result = compute_gate_score(vf, fp)
                val_scores.append(eval_result.gate_score)
                # CR-1: collect per-rule activations for w_i learning
                normalized = normalize_features(vf, fp.normalization)
                facts = compute_fact_activations(normalized)
                rules = compute_rule_activations(facts)
                val_rule_activations.append([strength for _, strength, _ in rules])

            rule_names = [name for name, _, _ in rules] if val_rule_activations else None
            t_low, t_high, w_i = calibrate_per_model(
                val_scores, val_labels,
                rule_names=rule_names,
                rule_activations=val_rule_activations,
            )

            # Update fingerprint with calibrated values
            fp.t_low = t_low
            fp.t_high = t_high
            fp.w_i = w_i
            fp.last_calibrated_at = datetime.now(timezone.utc).isoformat()

            with _calibration_lock:
                _fingerprints[model_id] = fp
                try:
                    FP_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(FP_STORAGE_DIR / f"{model_id}.json", "w") as f:
                        json.dump(fp.model_dump(), f, indent=2)
                except Exception:
                    pass
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
        finally:
            with _calibration_lock:
                _active_calibrations.discard(model_id)

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


# ── SPA / Static Files Serving ──

_dist_dir = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _dist_dir.exists():
    _assets_dir = _dist_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA index.html or static files."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        target = _dist_dir / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(_dist_dir / "index.html")

