"""Gate scoring — compute_gate_score() and classify().

Per final architecture: score.py is split into:
- compute_gate_score(): PURE function, returns GateEvaluation
- classify(): applies thresholds to produce verdict
- escalate_if_ambiguous(): separate function (M7)
"""
from __future__ import annotations

from backend.app.fingerprint.cluster import normalize_features
from backend.app.gate.rules import (
    compute_fact_activations,
    compute_rule_activations,
    get_triggered_patterns,
)
from backend.app.schemas import (
    Fingerprint,
    GateEvaluation,
    Thresholds,
)


class UncalibratedModelError(Exception):
    """Raised when scoring is attempted on a model without calibrated thresholds."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(
            f"Model '{model_id}' has not been calibrated. "
            "Run calibration before scoring."
        )


def compute_gate_score(
    raw_features: dict[str, float],
    fingerprint: Fingerprint,
    w_i: dict[str, float] | None = None,
    r_i_strategy: str = "lower",
) -> GateEvaluation:
    """Compute gate score G = Σ w_i * r_i. Pure function, no side effects.

    G is a quantity this system defines and calibrates — it is NOT a
    hallucination probability (per spec Section 5.5 correction).

    Args:
        raw_features: Un-normalized feature dict {H, S, C, E, D, M}.
        fingerprint: The model's fingerprint with normalization params.
        w_i: Per-rule weights. If None, uses fingerprint's w_i or uniform.
        r_i_strategy: How to extract r_i from rule antecedents.

    Returns:
        GateEvaluation with score, features, patterns (no verdict yet).
    """
    # Normalize features using this model's params
    normalized = normalize_features(raw_features, fingerprint.normalization)

    # Compute fact and rule activations
    fact_activations = compute_fact_activations(normalized)
    rule_activations = compute_rule_activations(fact_activations, r_i_strategy)
    triggered = get_triggered_patterns(rule_activations)

    # Determine weights
    if w_i is None:
        w_i = fingerprint.w_i
    if w_i is None:
        # Uniform weights as fallback
        n_rules = len(rule_activations)
        w_i = {name: 1.0 / max(n_rules, 1) for name, _, _ in rule_activations}

    # G = Σ w_i * r_i
    gate_score = sum(
        w_i.get(name, 0.0) * strength
        for name, strength, _ in rule_activations
    )

    # Clamp to [0, 1]
    gate_score = max(0.0, min(1.0, gate_score))

    return GateEvaluation(
        gate_score=gate_score,
        gate_verdict="UNCLASSIFIED",  # classify() applies thresholds
        thresholds=Thresholds(t_low=0.0, t_high=1.0),  # placeholder
        triggered_patterns=triggered,
        feature_breakdown=normalized,
        raw_features=raw_features,
        model_id=fingerprint.model_id,
        fingerprint_version=fingerprint.version,
    )


def classify(
    gate_evaluation: GateEvaluation,
    fingerprint: Fingerprint,
) -> GateEvaluation:
    """Apply thresholds to produce verdict.

    Per ADR A-H1: explicit None-check before threshold comparison.
    Per ADR A-M2: constrain 0.0 ≤ T_L < T_H ≤ 1.0.

    Verdict logic (spec Section 5.6):
        G < T_L       → LOW_RISK
        T_L ≤ G ≤ T_H → AMBIGUOUS (needs judge escalation)
        G > T_H       → HIGH_RISK
    """
    # ADR A-H1: explicit None-check
    if fingerprint.t_low is None or fingerprint.t_high is None:
        raise UncalibratedModelError(fingerprint.model_id)

    t_low = fingerprint.t_low
    t_high = fingerprint.t_high

    # ADR A-M2: bounds validation — ensure 0.0 ≤ T_L < T_H ≤ 1.0
    t_low = max(0.0, min(t_low, 0.95))   # cap at 0.95 so t_high can be > t_low
    t_high = max(0.0, min(t_high, 1.0))
    if t_low >= t_high:
        # Push t_low down rather than t_high up, to avoid the t_low=1.0 deadlock
        t_high = min(t_low + 0.05, 1.0)
        if t_low >= t_high:
            t_low = t_high - 0.05

    G = gate_evaluation.gate_score

    if G < t_low:
        verdict = "LOW_RISK"
    elif G > t_high:
        verdict = "HIGH_RISK"
    else:
        verdict = "AMBIGUOUS"

    return GateEvaluation(
        gate_score=G,
        gate_verdict=verdict,
        thresholds=Thresholds(t_low=t_low, t_high=t_high),
        triggered_patterns=gate_evaluation.triggered_patterns,
        feature_breakdown=gate_evaluation.feature_breakdown,
        raw_features=gate_evaluation.raw_features,
        model_id=gate_evaluation.model_id,
        fingerprint_version=gate_evaluation.fingerprint_version,
    )


def calibrate_global_thresholds(
    gate_scores: list[float],
    labels: list[int],
    target_escalation_rate: float = 0.3,
) -> tuple[float, float, dict[str, float]]:
    """Learn global (T_L, T_H) on pooled data.

    Simple strategy: set T_L and T_H based on score distribution
    to achieve approximately the target escalation rate.

    Returns: (t_low, t_high, w_i)
    """
    import numpy as np

    scores = np.array(gate_scores)
    labels_arr = np.array(labels)

    if len(scores) == 0:
        return 0.3, 0.7, {}

    # Sort scores and find thresholds that bracket the ambiguous zone
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    # T_L: threshold below which we're confident it's not hallucination
    # T_H: threshold above which we're confident it is hallucination
    # Target: ~target_escalation_rate of samples fall in [T_L, T_H]
    margin = target_escalation_rate / 2

    t_low_idx = int(n * (0.5 - margin))
    t_high_idx = int(n * (0.5 + margin))

    t_low = float(sorted_scores[max(0, t_low_idx)])
    t_high = float(sorted_scores[min(n - 1, t_high_idx)])

    # Ensure valid bounds
    if t_low >= t_high:
        t_high = min(t_low + 0.05, 1.0)

    # Uniform weights for global baseline (Baseline 3 per architecture)
    w_i = {
        "anomaly_pattern_1": 1.0 / 3,
        "anomaly_pattern_2": 1.0 / 3,
        "anomaly_pattern_3": 1.0 / 3,
    }

    return t_low, t_high, w_i
