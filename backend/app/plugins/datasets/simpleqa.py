"""SimpleQA dataset loader — secondary/evaluation-only dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from backend.app.schemas import DatasetRole, Record


class SimpleQALoader:
    """Loads SimpleQA from a CSV file.

    SimpleQA is used as a secondary evaluation dataset. Label mapping:
      - grade "CORRECT" → label=0
      - grade "NOT_ATTEMPTED" or "INCORRECT" → label=1

    All records are tagged dataset_role=SECONDARY.
    """

    name = "simpleqa"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load(self) -> Iterable[Record]:
        with open(self.data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("problem", "").strip()
                answer = row.get("answer", "").strip()
                if not question or not answer:
                    continue

                grade = row.get("grade", "").strip().upper()
                if grade == "CORRECT":
                    label = 0
                else:
                    label = 1

                model_id = row.get("model", "simpleqa_model").strip()

                yield Record(
                    question=question,
                    answer=answer,
                    model_id=model_id,
                    label=label,
                    source_dataset="simpleqa",
                    dataset_role=DatasetRole.SECONDARY,
                )
