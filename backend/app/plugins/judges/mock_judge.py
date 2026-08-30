"""Mock judge plugin — deterministic, no API key required.

Per spec: must exist and be the default in tests and local dev without
an API key, to avoid tests silently depending on a live API key.
"""
from __future__ import annotations

from backend.app.schemas import JudgeResult


class MockJudgePlugin:
    """Returns a deterministic canned response for testing."""

    name = "mock"

    def judge(
        self,
        answer: str,
        fingerprint_summary: dict,
        ambiguous_patterns: list[dict],
    ) -> JudgeResult:
        # Simple heuristic: if many patterns triggered, lean HIGH_RISK
        if len(ambiguous_patterns) >= 2:
            verdict = "HIGH_RISK"
            explanation = (
                "Multiple anomaly patterns were triggered simultaneously, "
                "suggesting systematic hallucination markers."
            )
            confidence = 0.8
        elif len(ambiguous_patterns) == 1:
            verdict = "LOW_RISK"
            explanation = (
                "Only one anomaly pattern was triggered, which is within "
                "normal variation for this type of response."
            )
            confidence = 0.6
        else:
            verdict = "LOW_RISK"
            explanation = "No significant anomaly patterns detected."
            confidence = 0.9

        return JudgeResult(
            verdict=verdict,
            explanation=explanation,
            confidence=confidence,
        )
