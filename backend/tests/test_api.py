"""Tests for FastAPI endpoints — M9."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import app, register_fingerprint, _fingerprints, _jobs
from backend.app.fingerprint.cluster import FEATURE_KEYS
from backend.app.schemas import (
    Fingerprint,
    NormalizationParams,
    ClusterInfo,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global state between tests and register plugins."""
    _fingerprints.clear()
    _jobs.clear()
    # Ensure plugins are registered (lifespan may not re-run per test)
    from backend.app.plugins.features.default import DefaultFeatureExtractor
    from backend.app.plugins.judges.mock_judge import MockJudgePlugin
    from backend.app.registry import register_feature_extractor, register_judge
    register_feature_extractor(DefaultFeatureExtractor())
    register_judge(MockJudgePlugin())
    yield
    _fingerprints.clear()
    _jobs.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _make_calibrated_fingerprint(model_id: str = "test_model") -> Fingerprint:
    return Fingerprint(
        model_id=model_id,
        version="v1",
        normalization=NormalizationParams(
            feature_mins={k: 0.0 for k in FEATURE_KEYS},
            feature_maxs={k: 10.0 for k in FEATURE_KEYS},
        ),
        clusters=ClusterInfo(
            centroids=[
                {k: 0.3 for k in FEATURE_KEYS},
                {k: 0.7 for k in FEATURE_KEYS},
            ],
            cluster_labels=["correct_region", "hallucination_region"],
        ),
        w_i={f"anomaly_pattern_{i}": 1.0 / 3 for i in range(1, 4)},
        t_low=0.3,
        t_high=0.7,
        calibration_dataset_size=100,
        last_calibrated_at="2026-08-30T00:00:00Z",
    )


# ── Health endpoint ──

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


# ── Score endpoint ──

class TestScore:
    def test_score_with_valid_fingerprint(self, client):
        fp = _make_calibrated_fingerprint()
        register_fingerprint(fp)

        resp = client.post("/api/v1/score", json={
            "answer": "Paris is the capital of France.",
            "model_id": "test_model",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Verify response schema matches spec Section 5.12
        assert "verdict" in data
        assert data["verdict"] in ("LOW_RISK", "HIGH_RISK", "RESOLVED_AMBIGUOUS")
        assert "gate_score" in data
        assert isinstance(data["gate_score"], float)
        assert "thresholds" in data
        assert "t_low" in data["thresholds"]
        assert "t_high" in data["thresholds"]
        assert "resolved_by" in data
        assert data["resolved_by"] in ("gate", "llm_judge")
        assert "triggered_patterns" in data
        assert "feature_breakdown" in data
        assert set(data["feature_breakdown"].keys()) == set(FEATURE_KEYS)
        assert "explanation" in data
        assert "explanation_source" in data
        assert data["explanation_source"] in ("symbolic", "llm_judge")
        assert "model_id" in data
        assert "fingerprint_version" in data

    def test_score_missing_model_returns_404(self, client):
        resp = client.post("/api/v1/score", json={
            "answer": "test",
            "model_id": "nonexistent_model",
        })
        assert resp.status_code == 404

    def test_score_missing_answer_returns_422(self, client):
        resp = client.post("/api/v1/score", json={
            "model_id": "test_model",
        })
        assert resp.status_code == 422


# ── Models endpoint ──

class TestModels:
    def test_list_models_empty(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_models_with_fingerprint(self, client):
        fp = _make_calibrated_fingerprint()
        register_fingerprint(fp)
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 1
        assert models[0]["model_id"] == "test_model"
        assert models[0]["fingerprint_version"] == "v1"
        assert models[0]["calibration_dataset_size"] == 100


# ── Fingerprint endpoint ──

class TestFingerprint:
    def test_get_fingerprint(self, client):
        fp = _make_calibrated_fingerprint()
        register_fingerprint(fp)
        resp = client.get("/api/v1/models/test_model/fingerprint")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "test_model"
        assert data["t_low"] == 0.3
        assert data["t_high"] == 0.7
        assert "clusters" in data
        assert "w_i" in data

    def test_get_fingerprint_missing_returns_404(self, client):
        resp = client.get("/api/v1/models/nonexistent/fingerprint")
        assert resp.status_code == 404


# ── Jobs endpoint ──

class TestJobs:
    def test_job_not_found_returns_404(self, client):
        resp = client.get("/api/v1/jobs/nonexistent-job-id")
        assert resp.status_code == 404


# ── Eval summary endpoint ──

class TestEvalSummary:
    def test_eval_summary_no_results(self, client):
        resp = client.get("/api/v1/eval/summary")
        assert resp.status_code == 200
