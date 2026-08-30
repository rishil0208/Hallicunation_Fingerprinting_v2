"""Escalation logic — bridges gate verdict with judge plugin.

Per final architecture: escalate_if_ambiguous() is separate from
compute_gate_score() to keep scoring pure and side-effect-free.
"""
from __future__ import annotations

from backend.app.gate.score import classify, compute_gate_score
from backend.app.plugins.judges.gemini_judge import GeminiJudgeError
from backend.app.registry import get_judge
from backend.app.schemas import Fingerprint, GateResult, Thresholds


def score_answer(
    answer: str,
    model_id: str,
    fingerprint: Fingerprint,
    feature_extractor_name: str = "default",
    judge_name: str = "mock",
    r_i_strategy: str = "lower",
) -> GateResult:
    """Full pipeline: extract features → score → classify → escalate if needed.

    This is the top-level function that the API endpoint calls.
    """
    from backend.app.registry import get_feature_extractor

    # Stage 1: Extract features
    extractor = get_feature_extractor(feature_extractor_name)
    raw_features = extractor.extract(answer)

    # Stage 2: Compute gate score and classify
    evaluation = compute_gate_score(raw_features, fingerprint, r_i_strategy=r_i_strategy)
    classified = classify(evaluation, fingerprint)

    # Stage 3: Escalate if ambiguous
    if classified.gate_verdict == "AMBIGUOUS":
        return escalate_if_ambiguous(classified, answer, fingerprint, judge_name)

    # Not ambiguous — return gate-only result
    explanation = _build_symbolic_explanation(classified.gate_verdict, classified.triggered_patterns)

    return GateResult(
        verdict=classified.gate_verdict,
        gate_score=classified.gate_score,
        thresholds=classified.thresholds,
        resolved_by="gate",
        triggered_patterns=classified.triggered_patterns,
        feature_breakdown=classified.feature_breakdown,
        explanation=explanation,
        explanation_source="symbolic",
        model_id=classified.model_id,
        fingerprint_version=classified.fingerprint_version,
    )


def escalate_if_ambiguous(
    classified_evaluation,
    answer: str,
    fingerprint: Fingerprint,
    judge_name: str = "mock",
) -> GateResult:
    """Escalate an AMBIGUOUS verdict to the LLM judge.

    ADR A-C1: If the judge fails, raises GeminiJudgeError.
    The API layer catches this and returns HTTP 502.
    """
    judge = get_judge(judge_name)

    # Build fingerprint summary for the judge (anonymized per ADR A-M1)
    fingerprint_summary = {
        "version": fingerprint.version,
        "gate_score": classified_evaluation.gate_score,
        "thresholds": {
            "t_low": classified_evaluation.thresholds.t_low,
            "t_high": classified_evaluation.thresholds.t_high,
        },
    }

    ambiguous_patterns = [
        {
            "name": p.name,
            "features_involved": p.features_involved,
            "strength": p.strength,
        }
        for p in classified_evaluation.triggered_patterns
    ]

    # ADR A-H3: assert patterns non-empty for AMBIGUOUS (invariant)
    assert len(ambiguous_patterns) > 0 or classified_evaluation.gate_score == 0, (
        "AMBIGUOUS verdict with G > 0 must have at least one triggered pattern"
    )

    # Call judge — may raise GeminiJudgeError (ADR A-C1)
    judge_result = judge.judge(answer, fingerprint_summary, ambiguous_patterns)

    return GateResult(
        verdict="RESOLVED_AMBIGUOUS",
        gate_score=classified_evaluation.gate_score,
        thresholds=classified_evaluation.thresholds,
        resolved_by="llm_judge",
        triggered_patterns=classified_evaluation.triggered_patterns,
        feature_breakdown=classified_evaluation.feature_breakdown,
        explanation=judge_result.explanation,
        explanation_source="llm_judge",
        model_id=classified_evaluation.model_id,
        fingerprint_version=classified_evaluation.fingerprint_version,
    )


def _build_symbolic_explanation(verdict: str, triggered_patterns) -> str:
    """Build a human-readable explanation from gate-only verdict."""
    if verdict == "LOW_RISK":
        return "The response shows low hallucination risk based on feature analysis."

    if verdict == "HIGH_RISK":
        pattern_names = [p.name for p in triggered_patterns]
        if pattern_names:
            return (
                f"High hallucination risk detected. Triggered patterns: "
                f"{', '.join(pattern_names)}."
            )
        return "High hallucination risk detected based on feature analysis."

    return "Analysis complete."
