"""Tests for data ingestion — M1."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.schemas import DatasetRole, Record
from backend.app.plugins.datasets.halueval_qa import HaluEvalQALoader
from backend.app.plugins.datasets.truthfulqa import TruthfulQALoader
from backend.app.plugins.datasets.simpleqa import SimpleQALoader
from backend.app.registry import (
    register_dataset_loader,
    get_dataset_loader,
    list_dataset_loaders,
)


# ── Record schema tests ──

class TestRecord:
    def test_record_requires_dataset_role(self):
        """dataset_role is required — construction without it must fail."""
        with pytest.raises(ValidationError):
            Record(
                question="Q",
                answer="A",
                model_id="test",
                label=0,
                source_dataset="test",
                # dataset_role deliberately omitted
            )

    def test_record_rejects_invalid_role(self):
        with pytest.raises(ValidationError):
            Record(
                question="Q",
                answer="A",
                model_id="test",
                label=0,
                source_dataset="test",
                dataset_role="calibration",  # not a valid role
            )

    def test_record_rejects_invalid_label(self):
        with pytest.raises(ValidationError):
            Record(
                question="Q",
                answer="A",
                model_id="test",
                label=2,  # must be 0 or 1
                source_dataset="test",
                dataset_role=DatasetRole.PRIMARY,
            )

    def test_record_accepts_valid_primary(self):
        r = Record(
            question="Q",
            answer="A",
            model_id="test",
            label=0,
            source_dataset="test",
            dataset_role=DatasetRole.PRIMARY,
        )
        assert r.dataset_role == DatasetRole.PRIMARY

    def test_record_accepts_valid_secondary(self):
        r = Record(
            question="Q",
            answer="A",
            model_id="test",
            label=1,
            source_dataset="test",
            dataset_role=DatasetRole.SECONDARY,
        )
        assert r.dataset_role == DatasetRole.SECONDARY


# ── HaluEval QA loader tests ──

def _make_halueval_file(entries: list[dict]) -> Path:
    """Write JSONL entries to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
    f.close()
    return Path(f.name)


HALUEVAL_SAMPLE = [
    {
        "knowledge": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "right_answer": "Paris",
        "hallucinated_answer": "London is the capital of France.",
    },
    {
        "knowledge": "Water boils at 100°C.",
        "question": "At what temperature does water boil?",
        "right_answer": "100 degrees Celsius",
        "hallucinated_answer": "Water boils at 50 degrees Celsius.",
    },
]


class TestHaluEvalQALoader:
    def test_yields_two_records_per_entry(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        assert len(records) == 4  # 2 entries × 2 records each

    def test_correct_labels(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        labels = [r.label for r in records]
        assert labels == [0, 1, 0, 1]

    def test_all_records_tagged_primary(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        assert all(r.dataset_role == DatasetRole.PRIMARY for r in records)

    def test_source_dataset_is_halueval_qa(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        assert all(r.source_dataset == "halueval_qa" for r in records)

    def test_default_model_id_is_chatgpt(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        assert all(r.model_id == "chatgpt" for r in records)

    def test_custom_model_id(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path, default_model_id="gpt-4")
        records = list(loader.load())
        assert all(r.model_id == "gpt-4" for r in records)

    def test_answer_content(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE[:1])
        loader = HaluEvalQALoader(path)
        records = list(loader.load())
        assert records[0].answer == "Paris"
        assert records[1].answer == "London is the capital of France."


# ── TruthfulQA loader tests ──

def _make_truthfulqa_csv(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    import csv
    fieldnames = ["Question", "Best Answer", "Incorrect Answers"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    f.close()
    return Path(f.name)


class TestTruthfulQALoader:
    def test_yields_correct_and_incorrect(self):
        path = _make_truthfulqa_csv([
            {
                "Question": "Is the Earth flat?",
                "Best Answer": "No",
                "Incorrect Answers": "Yes; Definitely yes",
            }
        ])
        loader = TruthfulQALoader(path)
        records = list(loader.load())
        assert len(records) == 3  # 1 correct + 2 incorrect

    def test_all_tagged_secondary(self):
        path = _make_truthfulqa_csv([
            {
                "Question": "Q?",
                "Best Answer": "A",
                "Incorrect Answers": "B",
            }
        ])
        loader = TruthfulQALoader(path)
        records = list(loader.load())
        assert all(r.dataset_role == DatasetRole.SECONDARY for r in records)

    def test_labels_correct(self):
        path = _make_truthfulqa_csv([
            {
                "Question": "Q?",
                "Best Answer": "A",
                "Incorrect Answers": "B; C",
            }
        ])
        loader = TruthfulQALoader(path)
        records = list(loader.load())
        labels = [r.label for r in records]
        assert labels == [0, 1, 1]


# ── SimpleQA loader tests ──

def _make_simpleqa_csv(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    import csv
    fieldnames = ["problem", "answer", "grade", "model"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    f.close()
    return Path(f.name)


class TestSimpleQALoader:
    def test_correct_grade_maps_to_label_0(self):
        path = _make_simpleqa_csv([
            {"problem": "Q?", "answer": "A", "grade": "CORRECT", "model": "m1"}
        ])
        loader = SimpleQALoader(path)
        records = list(loader.load())
        assert len(records) == 1
        assert records[0].label == 0

    def test_incorrect_grade_maps_to_label_1(self):
        path = _make_simpleqa_csv([
            {"problem": "Q?", "answer": "A", "grade": "INCORRECT", "model": "m1"}
        ])
        loader = SimpleQALoader(path)
        records = list(loader.load())
        assert records[0].label == 1

    def test_all_tagged_secondary(self):
        path = _make_simpleqa_csv([
            {"problem": "Q?", "answer": "A", "grade": "CORRECT", "model": "m1"}
        ])
        loader = SimpleQALoader(path)
        records = list(loader.load())
        assert all(r.dataset_role == DatasetRole.SECONDARY for r in records)


# ── Calibration leakage test ──

class TestCalibrationLeakage:
    """Verify that filtering by dataset_role works at the type level."""

    def test_primary_filter_excludes_secondary(self):
        """Simulates what calibration code must do: filter to PRIMARY only."""
        primary = Record(
            question="Q", answer="A", model_id="m",
            label=0, source_dataset="halueval_qa",
            dataset_role=DatasetRole.PRIMARY,
        )
        secondary = Record(
            question="Q", answer="A", model_id="m",
            label=1, source_dataset="truthfulqa",
            dataset_role=DatasetRole.SECONDARY,
        )
        all_records = [primary, secondary]

        # This is the filter calibration code must apply
        calibration_records = [
            r for r in all_records if r.dataset_role == DatasetRole.PRIMARY
        ]
        assert len(calibration_records) == 1
        assert calibration_records[0].dataset_role == DatasetRole.PRIMARY
        assert secondary not in calibration_records


# ── Registry tests ──

class TestRegistry:
    def test_register_and_retrieve_loader(self):
        path = _make_halueval_file(HALUEVAL_SAMPLE)
        loader = HaluEvalQALoader(path)
        register_dataset_loader(loader)
        retrieved = get_dataset_loader("halueval_qa")
        assert retrieved is loader

    def test_unknown_loader_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_dataset_loader("nonexistent")
