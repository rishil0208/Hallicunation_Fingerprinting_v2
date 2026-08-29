"""HaluEval QA dataset loader — primary dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from backend.app.schemas import DatasetRole, Record


class HaluEvalQALoader:
    name = "halueval_qa"

    def __init__(self, data_path: str | Path, default_model_id: str = "chatgpt"):
        self.data_path = Path(data_path)
        self.default_model_id = default_model_id

    def load(self) -> Iterable[Record]:
        """Load HaluEval QA subset.

        Each line in qa_data.json is a JSON object with:
          knowledge, question, right_answer, hallucinated_answer

        Yields two Records per line:
          - right_answer with label=0 (correct)
          - hallucinated_answer with label=1 (hallucinated)
        """
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                yield Record(
                    question=entry["question"],
                    answer=entry["right_answer"],
                    model_id=self.default_model_id,
                    label=0,
                    source_dataset="halueval_qa",
                    dataset_role=DatasetRole.PRIMARY,
                )
                yield Record(
                    question=entry["question"],
                    answer=entry["hallucinated_answer"],
                    model_id=self.default_model_id,
                    label=1,
                    source_dataset="halueval_qa",
                    dataset_role=DatasetRole.PRIMARY,
                )
