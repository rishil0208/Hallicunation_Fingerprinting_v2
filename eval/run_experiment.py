"""Experiment harness — runs baselines/models on evaluation data."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.schemas import DatasetRole, Record
from eval.metrics import auroc, auprc, ece, escalation_rate


def run_experiment(
    name: str,
    records: list[Record],
    predictor: Any,
    output_dir: str | Path = "eval/results",
    dataset_role_filter: DatasetRole | None = None,
) -> dict:
    """Run a single experiment and write results.

    Args:
        name: Experiment identifier.
        records: Dataset records.
        predictor: Object with .predict(records) -> list[dict].
        output_dir: Where to write JSON results.
        dataset_role_filter: If set, only use records with this role.

    Returns:
        Results dict with metrics.
    """
    if dataset_role_filter is not None:
        records = [r for r in records if r.dataset_role == dataset_role_filter]

    if not records:
        raise ValueError(f"No records after filtering for {dataset_role_filter}")

    predictions = predictor.predict(records)

    y_true = [r.label for r in records]
    y_score = [p["score"] for p in predictions]
    verdicts = [p["verdict"] for p in predictions]

    results = {
        "experiment": name,
        "predictor": getattr(predictor, "name", type(predictor).__name__),
        "n_records": len(records),
        "n_positive": sum(y_true),
        "n_negative": len(y_true) - sum(y_true),
        "metrics": {
            "auroc": auroc(y_true, y_score),
            "auprc": auprc(y_true, y_score),
            "ece": ece(y_true, y_score),
            "escalation_rate": escalation_rate(verdicts),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{name}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    return results
