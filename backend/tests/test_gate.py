"""Tests for gate scoring — M5."""
from __future__ import annotations

import pytest

from backend.app.fingerprint.cluster import build_fingerprint, FEATURE_KEYS
from backend.app.gate.rules import (
    compute_fact_activations,
    compute_rule_activations,
    get_triggered_patterns,
)
from backend.app.gate.score import (
    UncalibratedModelError,
    calibrate_global_thresholds,
    classify,
    compute_gate_score,
)
from backend.app.schemas import Fingerprint, NormalizationParams, Thresholds


def _make_calibrated_fingerprint(
    t_low: float = 0.3, t_high: float = 0.7,
) -> Fingerprint:
    """Create a minimal calibrated fingerprint for testing."""
    return Fingerprint(
        model_id="test_model",
        version="v1",
        normalization=NormalizationParams(
            feature_mins={k: 0.0 for k in FEATURE_KEYS},
            feature_maxs={k: 10.0 for k in FEATURE_KEYS},
        ),
        clusters=None,
        w_i={
            "anomaly_pattern_1": 1.0 / 3,
            "anomaly_pattern_2": 1.0 / 3,
            "anomaly_pattern_3": 1.0 / 3,
        },
        t_low=t_low,
        t_high=t_high,
    )


# ── Fact activation tests ──

class TestFactActivations:
    def test_high_hedge_activates_hedging_high(self):
        features = {"H": 0.8, "S": 0.5, "C": 0.0, "E": 0.5, "D": 0.0, "M": 0.0}
        activations = compute_fact_activations(features)
        assert activations["hedging_high"] > 0

    def test_low_hedge_does_not_activate(self):
        features = {"H": 0.1, "S": 0.5, "C": 0.0, "E": 0.5, "D": 0.0, "M": 0.0}
        activations = compute_fact_activations(features)
        assert activations["hedging_high"] == 0.0

    def test_low_specificity_activates_specificity_low(self):
        features = {"H": 0.0, "S": 0.2, "C": 0.0, "E": 0.5, "D": 0.0, "M": 0.0}
        activations = compute_fact_activations(features)
        assert activations["specificity_low"] > 0


# ── Rule activation tests ──

class TestRuleActivations:
    def test_pattern1_fires_with_high_hedge_and_vague_citation(self):
        facts = {
            "hedging_high": 0.8,
            "specificity_low": 0.0,
            "citation_vague": 0.7,
            "evidence_low": 0.0,
            "drift_high": 0.0,
            "confidence_high": 0.0,
        }
        rules = compute_rule_activations(facts)
        pattern1 = [r for r in rules if r[0] == "anomaly_pattern_1"]
        assert len(pattern1) == 1
        assert pattern1[0][1] > 0  # strength > 0

    def test_pattern1_does_not_fire_without_citation(self):
        facts = {
            "hedging_high": 0.8,
            "specificity_low": 0.0,
            "citation_vague": 0.0,  # not active
            "evidence_low": 0.0,
            "drift_high": 0.0,
            "confidence_high": 0.0,
        }
        rules = compute_rule_activations(facts)
        pattern1 = [r for r in rules if r[0] == "anomaly_pattern_1"]
        assert pattern1[0][1] == 0.0

    def test_lower_strategy_uses_min(self):
        facts = {
            "hedging_high": 0.8,
            "specificity_low": 0.0,
            "citation_vague": 0.3,
            "evidence_low": 0.0,
            "drift_high": 0.0,
            "confidence_high": 0.0,
        }
        rules = compute_rule_activations(facts, r_i_strategy="lower")
        pattern1 = [r for r in rules if r[0] == "anomaly_pattern_1"]
        assert pattern1[0][1] == 0.3  # min(0.8, 0.3)


# ── Gate scoring tests ──

class TestGateScoring:
    def test_gate_score_is_bounded_0_1(self):
        fp = _make_calibrated_fingerprint()
        raw = {"H": 8.0, "S": 1.0, "C": 5.0, "E": 1.0, "D": 5.0, "M": 8.0}
        result = compute_gate_score(raw, fp)
        assert 0.0 <= result.gate_score <= 1.0

    def test_benign_input_has_low_score(self):
        fp = _make_calibrated_fingerprint()
        raw = {"H": 0.0, "S": 8.0, "C": 0.0, "E": 8.0, "D": 0.0, "M": 0.0}
        result = compute_gate_score(raw, fp)
        assert result.gate_score < 0.5

    def test_suspicious_input_has_higher_score(self):
        fp = _make_calibrated_fingerprint()
        raw_benign = {"H": 0.0, "S": 8.0, "C": 0.0, "E": 8.0, "D": 0.0, "M": 0.0}
        raw_suspicious = {"H": 8.0, "S": 1.0, "C": 5.0, "E": 1.0, "D": 5.0, "M": 8.0}
        benign = compute_gate_score(raw_benign, fp)
        suspicious = compute_gate_score(raw_suspicious, fp)
        assert suspicious.gate_score >= benign.gate_score

    def test_returns_all_features(self):
        fp = _make_calibrated_fingerprint()
        raw = {"H": 5.0, "S": 5.0, "C": 5.0, "E": 5.0, "D": 5.0, "M": 5.0}
        result = compute_gate_score(raw, fp)
        assert set(result.raw_features.keys()) == set(FEATURE_KEYS)
        assert set(result.feature_breakdown.keys()) == set(FEATURE_KEYS)


# ── Classify tests ──

class TestClassify:
    def test_low_risk_verdict(self):
        fp = _make_calibrated_fingerprint(t_low=0.3, t_high=0.7)
        raw = {"H": 0.0, "S": 8.0, "C": 0.0, "E": 8.0, "D": 0.0, "M": 0.0}
        evaluation = compute_gate_score(raw, fp)
        result = classify(evaluation, fp)
        assert result.gate_verdict == "LOW_RISK"

    def test_uncalibrated_model_raises(self):
        fp = Fingerprint(
            model_id="uncal",
            version="v1",
            normalization=NormalizationParams(
                feature_mins={k: 0.0 for k in FEATURE_KEYS},
                feature_maxs={k: 10.0 for k in FEATURE_KEYS},
            ),
            clusters=None,
            t_low=None,
            t_high=None,
        )
        raw = {"H": 5.0, "S": 5.0, "C": 5.0, "E": 5.0, "D": 5.0, "M": 5.0}
        evaluation = compute_gate_score(raw, fp)
        with pytest.raises(UncalibratedModelError, match="uncal"):
            classify(evaluation, fp)

    def test_threshold_bounds_enforced(self):
        """ADR A-M2: T_L >= T_H gets corrected to T_H = T_L + 0.05."""
        fp = _make_calibrated_fingerprint(t_low=0.5, t_high=0.5)
        raw = {"H": 5.0, "S": 5.0, "C": 5.0, "E": 5.0, "D": 5.0, "M": 5.0}
        evaluation = compute_gate_score(raw, fp)
        result = classify(evaluation, fp)
        assert result.thresholds.t_low < result.thresholds.t_high


# ── Global calibration tests ──

class TestGlobalCalibration:
    def test_produces_valid_thresholds(self):
        scores = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        t_low, t_high, w_i = calibrate_global_thresholds(scores, labels)
        assert 0.0 <= t_low < t_high <= 1.0
        assert len(w_i) == 3

    def test_empty_scores_returns_defaults(self):
        t_low, t_high, w_i = calibrate_global_thresholds([], [])
        assert t_low == 0.3
        assert t_high == 0.7


# ── Triggered patterns test ──

class TestTriggeredPatterns:
    def test_triggered_patterns_non_empty_for_anomalous_input(self):
        """ADR A-H3: triggered_patterns cannot be empty when G > 0."""
        fp = _make_calibrated_fingerprint()
        raw = {"H": 8.0, "S": 1.0, "C": 5.0, "E": 1.0, "D": 5.0, "M": 8.0}
        result = compute_gate_score(raw, fp)
        if result.gate_score > 0:
            assert len(result.triggered_patterns) > 0
