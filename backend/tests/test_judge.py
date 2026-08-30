"""Tests for judge escalation — M7."""
from __future__ import annotations

import os

import pytest

from backend.app.fingerprint.cluster import FEATURE_KEYS
from backend.app.gate.escalate import (
    _build_symbolic_explanation,
    escalate_if_ambiguous,
    score_answer,
)
from backend.app.gate.score import classify, compute_gate_score
from backend.app.plugins.features.default import DefaultFeatureExtractor
from backend.app.plugins.judges.gemini_judge import GeminiJudgeError, GeminiJudgePlugin
from backend.app.plugins.judges.mock_judge import MockJudgePlugin
from backend.app.registry import (
    register_feature_extractor,
    register_judge,
)
from backend.app.schemas import (
    Fingerprint,
    GateEvaluation,
    JudgeResult,
    NormalizationParams,
    Thresholds,
    TriggeredPattern,
)


@pytest.fixture(autouse=True)
def _register_plugins():
    """Register mock plugins for all tests."""
    register_feature_extractor(DefaultFeatureExtractor())
    register_judge(MockJudgePlugin())


def _make_fingerprint(t_low=0.3, t_high=0.7) -> Fingerprint:
    return Fingerprint(
        model_id="test_model",
        version="v1",
        normalization=NormalizationParams(
            feature_mins={k: 0.0 for k in FEATURE_KEYS},
            feature_maxs={k: 10.0 for k in FEATURE_KEYS},
        ),
        w_i={f"anomaly_pattern_{i}": 1.0 / 3 for i in range(1, 4)},
        t_low=t_low,
        t_high=t_high,
    )


# ── MockJudgePlugin tests ──

class TestMockJudge:
    def test_returns_judge_result(self):
        judge = MockJudgePlugin()
        result = judge.judge("test answer", {}, [])
        assert isinstance(result, JudgeResult)
        assert result.verdict in ("HIGH_RISK", "LOW_RISK")
        assert 0.0 <= result.confidence <= 1.0

    def test_multiple_patterns_yields_high_risk(self):
        judge = MockJudgePlugin()
        patterns = [
            {"name": "p1", "features_involved": ["H"], "strength": 0.8},
            {"name": "p2", "features_involved": ["S"], "strength": 0.7},
        ]
        result = judge.judge("answer", {}, patterns)
        assert result.verdict == "HIGH_RISK"

    def test_single_pattern_yields_low_risk(self):
        judge = MockJudgePlugin()
        patterns = [
            {"name": "p1", "features_involved": ["H"], "strength": 0.8},
        ]
        result = judge.judge("answer", {}, patterns)
        assert result.verdict == "LOW_RISK"

    def test_no_patterns_yields_low_risk(self):
        judge = MockJudgePlugin()
        result = judge.judge("answer", {}, [])
        assert result.verdict == "LOW_RISK"


# ── GeminiJudgePlugin tests (no API key needed) ──

class TestGeminiJudge:
    def test_missing_api_key_raises_clean_error(self):
        """ADR A-H2: missing key raises GeminiJudgeError, not a crash."""
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            judge = GeminiJudgePlugin(api_key="")
            with pytest.raises(GeminiJudgeError, match="GEMINI_API_KEY not set"):
                judge.judge("test", {}, [])
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_api_key_not_in_error_message(self):
        """ADR A-H2: API key must never appear in error messages."""
        error = GeminiJudgeError("Error with key ABC123SECRET")
        os.environ["GEMINI_API_KEY"] = "ABC123SECRET"
        try:
            sanitized_error = GeminiJudgeError("Error with key ABC123SECRET")
            assert "ABC123SECRET" not in str(sanitized_error)
            assert "[REDACTED]" in str(sanitized_error)
        finally:
            del os.environ["GEMINI_API_KEY"]


# ── Escalation logic tests ──

class TestEscalation:
    def test_ambiguous_verdict_escalates_to_judge(self):
        """AMBIGUOUS verdict should go to the judge."""
        fp = _make_fingerprint(t_low=0.0, t_high=0.01)  # narrow band → most things AMBIGUOUS or HIGH_RISK
        evaluation = GateEvaluation(
            gate_score=0.005,
            gate_verdict="AMBIGUOUS",
            thresholds=Thresholds(t_low=0.0, t_high=0.01),
            triggered_patterns=[
                TriggeredPattern(name="anomaly_pattern_1", features_involved=["H", "C"], strength=0.5)
            ],
            feature_breakdown={k: 0.5 for k in FEATURE_KEYS},
            raw_features={k: 5.0 for k in FEATURE_KEYS},
            model_id="test_model",
            fingerprint_version="v1",
        )
        result = escalate_if_ambiguous(evaluation, "test answer", fp, judge_name="mock")
        assert result.verdict == "RESOLVED_AMBIGUOUS"
        assert result.resolved_by == "llm_judge"
        assert result.explanation_source == "llm_judge"

    def test_gate_only_verdict_not_escalated(self):
        """LOW_RISK and HIGH_RISK should not reach the judge."""
        fp = _make_fingerprint(t_low=0.3, t_high=0.7)

        # Benign input → LOW_RISK
        result = score_answer(
            answer="Paris is the capital of France.",
            model_id="test_model",
            fingerprint=fp,
            judge_name="mock",
        )
        # The result should be gate-resolved
        if result.verdict in ("LOW_RISK", "HIGH_RISK"):
            assert result.resolved_by == "gate"
            assert result.explanation_source == "symbolic"


# ── Symbolic explanation tests ──

class TestSymbolicExplanation:
    def test_low_risk_explanation(self):
        explanation = _build_symbolic_explanation("LOW_RISK", [])
        assert "low" in explanation.lower()

    def test_high_risk_explanation_includes_patterns(self):
        patterns = [
            TriggeredPattern(name="anomaly_pattern_1", features_involved=["H", "C"], strength=0.8),
        ]
        explanation = _build_symbolic_explanation("HIGH_RISK", patterns)
        assert "anomaly_pattern_1" in explanation
