"""Alternate feature extractor — demonstrates the plugin interface is real.

Uses a different, smaller hedge/confidence lexicon to show that swapping
extractors produces different (but structurally identical) output.
"""
from __future__ import annotations

from backend.app.plugins.features.default import (
    _count_words,
    _specificity,
    _citation_vagueness,
    _evidence_density,
    _semantic_drift,
)

ALT_HEDGE_PHRASES = [
    "might", "maybe", "perhaps", "possibly", "I think", "it seems",
    "probably", "not sure", "approximately", "sort of",
]

ALT_CONFIDENCE_PHRASES = [
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "always", "never", "exactly", "must be",
]


def _alt_hedge_density(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for phrase in ALT_HEDGE_PHRASES if phrase in text_lower)
    words = _count_words(text)
    return (count / words) * 100


def _alt_confidence_density(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for phrase in ALT_CONFIDENCE_PHRASES if phrase in text_lower)
    words = _count_words(text)
    return (count / words) * 100


class AltFeatureExtractor:
    """Alternate lexicon for H and M; S, C, E, D remain the same."""

    name = "alt"

    def extract(self, answer: str) -> dict[str, float]:
        if not answer.strip():
            return {"H": 0.0, "S": 0.0, "C": 0.0, "E": 0.0, "D": 0.0, "M": 0.0}

        return {
            "H": _alt_hedge_density(answer),
            "S": _specificity(answer),
            "C": _citation_vagueness(answer),
            "E": _evidence_density(answer),
            "D": _semantic_drift(answer),
            "M": _alt_confidence_density(answer),
        }
