"""Baseline classifiers — spec Section 6 baselines 1 and 4."""
from __future__ import annotations

import random
from typing import Callable

from backend.app.schemas import Record


class AlwaysEscalateBaseline:
    """Baseline 1: escalate every answer to the LLM judge.

    For metric computation: assigns score=0.5 (uninformative) and
    verdict=RESOLVED_AMBIGUOUS for every record.
    """

    name = "always_escalate"

    def predict(self, records: list[Record]) -> list[dict]:
        return [
            {"score": 0.5, "verdict": "RESOLVED_AMBIGUOUS"}
            for _ in records
        ]


class RandomBaseline:
    """Baseline 4: random/majority-class classifier.

    Generates random scores from U(0,1) and assigns verdict based on
    a threshold of 0.5. Uses a fixed seed for reproducibility.
    """

    name = "random_baseline"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def predict(self, records: list[Record]) -> list[dict]:
        rng = random.Random(self.seed)
        results = []
        for _ in records:
            score = rng.random()
            if score >= 0.5:
                verdict = "HIGH_RISK"
            else:
                verdict = "LOW_RISK"
            results.append({"score": score, "verdict": verdict})
        return results


class MajorityClassBaseline:
    """Baseline 4 variant: always predicts the majority class."""

    name = "majority_class"

    def predict(self, records: list[Record]) -> list[dict]:
        # Determine majority label
        if not records:
            return []
        label_counts = {0: 0, 1: 0}
        for r in records:
            label_counts[r.label] += 1

        majority_label = max(label_counts, key=label_counts.get)
        score = float(majority_label)

        return [
            {"score": score, "verdict": "HIGH_RISK" if majority_label == 1 else "LOW_RISK"}
            for _ in records
        ]
