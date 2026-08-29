"""Calibration — split management and threshold fitting."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from backend.app.schemas import DatasetRole, Record


def split_records(
    records: list[Record],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Split records into train/val/test sets.

    Per spec: 60/20/20 default split. Only operates on records with
    dataset_role == PRIMARY. This is enforced at the call site, not
    here, to keep the function general-purpose.
    """
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n = len(records)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]

    return train, val, test


def ensure_primary_only(records: list[Record]) -> list[Record]:
    """Runtime assertion: calibration operates only on primary data.

    Per ADR and spec Section 5.1: secondary data must never leak into
    calibration. This function filters and warns.
    """
    primary = [r for r in records if r.dataset_role == DatasetRole.PRIMARY]
    n_filtered = len(records) - len(primary)
    if n_filtered > 0:
        import warnings
        warnings.warn(
            f"Filtered {n_filtered} non-primary records from calibration data. "
            "Calibration must use primary data only."
        )
    if not primary:
        raise ValueError("No primary records available for calibration")
    return primary


def save_split_indices(
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    output_dir: str | Path,
    model_id: str,
) -> Path:
    """Persist split indices for reproducibility."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_file = output_path / f"{model_id}_splits.json"
    with open(split_file, "w") as f:
        json.dump({
            "model_id": model_id,
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }, f, indent=2)
    return split_file
