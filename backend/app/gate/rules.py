"""Rule definitions — maps normalized features to PyReason-style rule activations."""
from __future__ import annotations

from typing import Optional

from backend.app.schemas import TriggeredPattern

# Default thresholds for feature → fact conversion
# These are the calibrated interval boundaries from spec Section 5.5
DEFAULT_FACT_THRESHOLDS = {
    "hedging_high": ("H", 0.5),       # normalized H >= 0.5 → hedging_high
    "specificity_high": ("S", 0.5),    # normalized S >= 0.5 → specificity_high (spec 5.5)
    "citation_vague": ("C", 0.3),      # normalized C >= 0.3 → citation_vague
    "evidence_low": ("E", 0.5),        # normalized E <= 0.5 → evidence_low (inverted)
    "drift_high": ("D", 0.3),          # normalized D >= 0.3 → drift_high
    "confidence_high": ("M", 0.5),     # normalized M >= 0.5 → confidence_high
}

# Relational pattern rules from spec Section 5.5
RULES = [
    {
        "name": "anomaly_pattern_1",
        "description": "High hedging + vague citations",
        "antecedents": ["hedging_high", "citation_vague"],
        "features_involved": ["H", "C"],
    },
    {
        "name": "anomaly_pattern_2",
        "description": "High specificity + low evidence (fabricated details)",
        "antecedents": ["specificity_high", "evidence_low"],
        "features_involved": ["S", "E"],
    },
    {
        "name": "anomaly_pattern_3",
        "description": "High confidence + high semantic drift",
        "antecedents": ["confidence_high", "drift_high"],
        "features_involved": ["M", "D"],
    },
]


def compute_fact_activations(
    normalized_features: dict[str, float],
    thresholds: dict | None = None,
) -> dict[str, float]:
    """Convert normalized features to fact activation strengths.

    Each fact has a strength in [0, 1] based on how far the feature
    exceeds (or falls below for inverted features) the threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_FACT_THRESHOLDS

    activations = {}
    for fact_name, (feature_key, threshold) in thresholds.items():
        val = normalized_features.get(feature_key, 0.0)

        # Inverted features: evidence_low (fires when value is BELOW threshold)
        if fact_name == "evidence_low":
            if val <= threshold:
                activations[fact_name] = 1.0 - (val / max(threshold, 1e-10))
            else:
                activations[fact_name] = 0.0
        else:
            if val >= threshold:
                activations[fact_name] = (val - threshold) / max(1.0 - threshold, 1e-10)
            else:
                activations[fact_name] = 0.0

    return activations


def compute_rule_activations(
    fact_activations: dict[str, float],
    r_i_strategy: str = "lower",
) -> list[tuple[str, float, list[str]]]:
    """Compute rule activation strengths from fact activations.

    A rule fires if ALL its antecedent facts are active (> 0).
    The rule's activation strength is determined by r_i_strategy:
      - "lower": min of antecedent strengths (conservative)
      - "upper": max of antecedent strengths
      - "midpoint": mean of antecedent strengths

    Returns: list of (rule_name, activation_strength, features_involved)
    """
    results = []
    for rule in RULES:
        antecedent_strengths = [
            fact_activations.get(ant, 0.0) for ant in rule["antecedents"]
        ]

        # Rule fires only if all antecedents are active
        if all(s > 0.0 for s in antecedent_strengths):
            if r_i_strategy == "lower":
                strength = min(antecedent_strengths)
            elif r_i_strategy == "upper":
                strength = max(antecedent_strengths)
            elif r_i_strategy == "midpoint":
                strength = sum(antecedent_strengths) / len(antecedent_strengths)
            else:
                strength = min(antecedent_strengths)
        else:
            strength = 0.0

        results.append((rule["name"], strength, rule["features_involved"]))

    return results


def get_triggered_patterns(
    rule_activations: list[tuple[str, float, list[str]]],
) -> list[TriggeredPattern]:
    """Convert active rule results to TriggeredPattern objects."""
    return [
        TriggeredPattern(
            name=name,
            features_involved=features,
            strength=strength,
        )
        for name, strength, features in rule_activations
        if strength > 0.0
    ]
