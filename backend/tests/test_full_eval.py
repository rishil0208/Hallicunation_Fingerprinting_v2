"""Tests for full evaluation — M8."""
from __future__ import annotations

import tempfile

import pytest

from backend.app.schemas import DatasetRole, Record
from eval.baselines import (
    AlwaysEscalateBaseline,
    LexicalBaseline,
    MajorityClassBaseline,
    NeverEscalateBaseline,
    RandomBaseline,
)
from eval.run_experiment import run_experiment


def _make_records(n: int = 100) -> list[Record]:
    """Create balanced test records with realistic text."""
    records = []
    for i in range(n // 2):
        records.append(Record(
            question=f"What is fact {i}?",
            answer=f"The answer is precisely {i}. This is verified.",
            model_id="test",
            label=0,
            source_dataset="test",
            dataset_role=DatasetRole.PRIMARY,
        ))
    for i in range(n // 2):
        records.append(Record(
            question=f"What is fact {n // 2 + i}?",
            answer=f"I think maybe the answer might possibly be {i}. Perhaps.",
            model_id="test",
            label=1,
            source_dataset="test",
            dataset_role=DatasetRole.PRIMARY,
        ))
    return records


class TestAllBaselinesExist:
    """Verify all 5 baselines from spec Section 6 are implemented."""

    def test_baseline_1_always_escalate(self):
        b = AlwaysEscalateBaseline()
        assert b.name == "always_escalate"

    def test_baseline_2_never_escalate(self):
        b = NeverEscalateBaseline()
        assert b.name == "never_escalate"

    def test_baseline_4_random(self):
        b = RandomBaseline()
        assert b.name == "random_baseline"

    def test_baseline_4_majority(self):
        b = MajorityClassBaseline()
        assert b.name == "majority_class"

    def test_baseline_5_lexical(self):
        b = LexicalBaseline()
        assert b.name == "lexical_baseline"


class TestBaselineBehavior:
    def test_never_escalate_never_produces_ambiguous(self):
        records = _make_records()
        baseline = NeverEscalateBaseline()
        preds = baseline.predict(records)
        assert all(p["verdict"] != "RESOLVED_AMBIGUOUS" for p in preds)

    def test_always_escalate_100_percent_escalation(self):
        records = _make_records()
        baseline = AlwaysEscalateBaseline()
        preds = baseline.predict(records)
        assert all(p["verdict"] == "RESOLVED_AMBIGUOUS" for p in preds)

    def test_lexical_baseline_scores_are_bounded(self):
        records = _make_records()
        baseline = LexicalBaseline()
        preds = baseline.predict(records)
        assert all(0.0 <= p["score"] <= 1.0 for p in preds)

    def test_lexical_detects_hedging_in_hallucinated(self):
        """Hallucinated answers (with hedge words) should score higher."""
        records = _make_records()
        baseline = LexicalBaseline()
        preds = baseline.predict(records)
        correct_scores = [p["score"] for p, r in zip(preds, records) if r.label == 0]
        halluc_scores = [p["score"] for p, r in zip(preds, records) if r.label == 1]
        # On average, hallucinated answers (which have hedge words) should score higher
        assert sum(halluc_scores) / len(halluc_scores) > sum(correct_scores) / len(correct_scores)


class TestHarnessIntegration:
    def test_all_baselines_produce_results(self):
        """Run all 5 baselines through the harness and verify output."""
        records = _make_records()
        baselines = [
            AlwaysEscalateBaseline(),
            NeverEscalateBaseline(),
            RandomBaseline(),
            MajorityClassBaseline(),
            LexicalBaseline(),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for baseline in baselines:
                results = run_experiment(
                    name=f"test_{baseline.name}",
                    records=records,
                    predictor=baseline,
                    output_dir=tmpdir,
                )
                assert "metrics" in results
                assert results["n_records"] == 100
                assert "auroc" in results["metrics"]
                assert "auprc" in results["metrics"]
                assert "ece" in results["metrics"]
                assert "escalation_rate" in results["metrics"]
