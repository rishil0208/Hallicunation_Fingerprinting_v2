"""Tests for eval harness and baselines — M3."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from backend.app.schemas import DatasetRole, Record
from eval.metrics import auroc, auprc, ece, escalation_rate
from eval.baselines import AlwaysEscalateBaseline, RandomBaseline, MajorityClassBaseline
from eval.run_experiment import run_experiment


def _make_records(n_pos: int = 50, n_neg: int = 50) -> list[Record]:
    """Create balanced test records."""
    records = []
    for i in range(n_neg):
        records.append(Record(
            question=f"Q{i}", answer=f"correct_{i}", model_id="test",
            label=0, source_dataset="test", dataset_role=DatasetRole.PRIMARY,
        ))
    for i in range(n_pos):
        records.append(Record(
            question=f"Q{n_neg + i}", answer=f"hallucinated_{i}", model_id="test",
            label=1, source_dataset="test", dataset_role=DatasetRole.PRIMARY,
        ))
    return records


# ── Metrics tests ──

class TestMetrics:
    def test_auroc_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert auroc(y_true, y_score) == 1.0

    def test_auroc_random_classifier(self):
        y_true = [0, 1, 0, 1]
        y_score = [0.5, 0.5, 0.5, 0.5]
        assert auroc(y_true, y_score) == 0.5

    def test_auroc_single_class_returns_nan(self):
        y_true = [0, 0, 0]
        y_score = [0.1, 0.2, 0.3]
        assert math.isnan(auroc(y_true, y_score))

    def test_auprc_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert auprc(y_true, y_score) == 1.0

    def test_ece_perfectly_calibrated(self):
        # If predicted probabilities match actual frequency, ECE ≈ 0
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_score = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert ece(y_true, y_score) < 0.01

    def test_ece_poorly_calibrated(self):
        # All predictions = 0.9 but only 50% are positive → poorly calibrated
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_score = [0.9] * 10
        assert ece(y_true, y_score) > 0.3

    def test_escalation_rate_all_escalated(self):
        verdicts = ["RESOLVED_AMBIGUOUS"] * 10
        assert escalation_rate(verdicts) == 1.0

    def test_escalation_rate_none_escalated(self):
        verdicts = ["LOW_RISK", "HIGH_RISK", "LOW_RISK"]
        assert escalation_rate(verdicts) == 0.0

    def test_escalation_rate_partial(self):
        verdicts = ["LOW_RISK", "RESOLVED_AMBIGUOUS", "HIGH_RISK", "RESOLVED_AMBIGUOUS"]
        assert escalation_rate(verdicts) == 0.5


# ── Baselines tests ──

class TestBaselines:
    def test_always_escalate_all_ambiguous(self):
        records = _make_records(10, 10)
        baseline = AlwaysEscalateBaseline()
        preds = baseline.predict(records)
        assert all(p["verdict"] == "RESOLVED_AMBIGUOUS" for p in preds)
        assert all(p["score"] == 0.5 for p in preds)

    def test_random_baseline_reproducible(self):
        records = _make_records(10, 10)
        b1 = RandomBaseline(seed=42)
        b2 = RandomBaseline(seed=42)
        p1 = b1.predict(records)
        p2 = b2.predict(records)
        assert [p["score"] for p in p1] == [p["score"] for p in p2]

    def test_random_baseline_different_seeds_differ(self):
        records = _make_records(10, 10)
        b1 = RandomBaseline(seed=42)
        b2 = RandomBaseline(seed=99)
        p1 = b1.predict(records)
        p2 = b2.predict(records)
        assert [p["score"] for p in p1] != [p["score"] for p in p2]

    def test_majority_class_predicts_majority(self):
        # 80 positive, 20 negative → majority is 1
        records = _make_records(n_pos=80, n_neg=20)
        baseline = MajorityClassBaseline()
        preds = baseline.predict(records)
        assert all(p["verdict"] == "HIGH_RISK" for p in preds)

    def test_majority_class_balanced(self):
        records = _make_records(n_pos=50, n_neg=50)
        baseline = MajorityClassBaseline()
        preds = baseline.predict(records)
        # With equal counts, either class could win; just check consistency
        assert len(set(p["verdict"] for p in preds)) == 1


# ── Experiment harness tests ──

class TestRunExperiment:
    def test_harness_produces_valid_results(self):
        records = _make_records(50, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_experiment(
                name="test_baseline1",
                records=records,
                predictor=AlwaysEscalateBaseline(),
                output_dir=tmpdir,
            )
            assert "metrics" in results
            assert "auroc" in results["metrics"]
            assert "auprc" in results["metrics"]
            assert "ece" in results["metrics"]
            assert "escalation_rate" in results["metrics"]
            assert results["metrics"]["escalation_rate"] == 1.0

            # Verify file was written
            result_file = Path(tmpdir) / "test_baseline1.json"
            assert result_file.exists()

    def test_harness_filters_by_dataset_role(self):
        primary = _make_records(10, 10)
        secondary = [Record(
            question="SQ", answer="SA", model_id="other",
            label=0, source_dataset="other",
            dataset_role=DatasetRole.SECONDARY,
        )]
        all_records = primary + secondary

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_experiment(
                name="test_filter",
                records=all_records,
                predictor=RandomBaseline(),
                output_dir=tmpdir,
                dataset_role_filter=DatasetRole.PRIMARY,
            )
            assert results["n_records"] == 20  # only primary records
