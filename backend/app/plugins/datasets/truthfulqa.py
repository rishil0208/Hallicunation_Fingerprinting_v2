"""TruthfulQA dataset loader — secondary/evaluation-only dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from backend.app.schemas import DatasetRole, Record


class TruthfulQALoader:
    """Loads TruthfulQA from a CSV file.

    TruthfulQA labels reflect factuality/misconception errors, not pure
    hallucination-as-fabrication. This is documented per spec Section 5.2
    and Section 6 rationale. All records are tagged dataset_role=SECONDARY.
    """

    name = "truthfulqa"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load(self) -> Iterable[Record]:
        with open(self.data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"]

                # Best correct answer → label=0
                best_answer = row.get("Best Answer", "").strip()
                if best_answer:
                    yield Record(
                        question=question,
                        answer=best_answer,
                        model_id="truthfulqa_reference",
                        label=0,
                        source_dataset="truthfulqa",
                        dataset_role=DatasetRole.SECONDARY,
                    )

                # Incorrect answers (semicolon-separated) → label=1
                # Note: these are factuality-proxy labels, not clean
                # hallucination labels (see spec Section 5.2 / Section 6).
                incorrect_raw = row.get("Incorrect Answers", "").strip()
                if incorrect_raw:
                    for ans in incorrect_raw.split(";"):
                        ans = ans.strip()
                        if ans:
                            yield Record(
                                question=question,
                                answer=ans,
                                model_id="truthfulqa_reference",
                                label=1,
                                source_dataset="truthfulqa",
                                dataset_role=DatasetRole.SECONDARY,
                            )
