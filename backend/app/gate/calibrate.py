"""Calibration — split management and threshold fitting."""
from __future__ import annotations

import json
import random
import warnings
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


def calibrate_per_model(
    gate_scores: list[float],
    labels: list[int],
    rule_names: list[str] | None = None,
    rule_activations: list[list[float]] | None = None,
    target_escalation_rate: float = 0.3,
) -> tuple[float, float, dict[str, float]]:
    """Learn per-model (T_L, T_H, w_i) — the core contribution.

    Per spec Section 5.6: w_i and per-model (T_L, T_H) are learned on
    a validation split. This uses a grid search over threshold pairs to
    maximize AUPRC while staying near the target escalation rate.

    CR-1 fix: w_i is learned from point-biserial correlation of each
    rule activation with ground-truth labels, then normalized to sum
    to 1.0. Gate scores are recomputed with learned weights.

    Constraint (ADR A-M2): 0.0 ≤ T_L < T_H ≤ 1.0.

    Args:
        gate_scores: Pre-computed G scores (used as fallback if no rule_activations).
        labels: Ground-truth binary labels (0=correct, 1=hallucinated).
        rule_names: Names for each rule dimension.
        rule_activations: Per-sample rule activation vectors [[r1, r2, r3], ...].
            If provided, w_i is learned from data and G is recomputed.
        target_escalation_rate: Target fraction of AMBIGUOUS verdicts.

    Returns: (t_low, t_high, w_i)
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    labels_arr = np.array(labels)

    if len(labels_arr) == 0:
        return 0.3, 0.7, {}

    # ── CR-1: Learn w_i from rule activations ──
    default_rules = rule_names or [
        "anomaly_pattern_1", "anomaly_pattern_2", "anomaly_pattern_3",
    ]

    if rule_activations is not None and len(rule_activations) > 0:
        R = np.array(rule_activations)  # shape (N, n_rules)
        n_rules = R.shape[1]

        # Point-biserial correlation: weight each rule by its
        # correlation with the positive (hallucinated) class
        raw_weights = []
        for j in range(n_rules):
            col = R[:, j]
            if col.std() < 1e-10:
                raw_weights.append(0.0)
            else:
                corr = np.corrcoef(col, labels_arr)[0, 1]
                raw_weights.append(max(0.0, corr))  # only positive correlations

        total = sum(raw_weights)
        if total > 1e-10:
            w_i = {default_rules[j]: raw_weights[j] / total for j in range(n_rules)}
        else:
            # When no rule exhibits positive correlation with hallucination labels,
            # do NOT silently reinstate uniform weights that would activate negatively correlated rules.
            # Set weights to 0.0, reflecting an uninformative gate for this data profile.
            w_i = {rn: 0.0 for rn in default_rules}
            warnings.warn(
                "No symbolic rule exhibited positive correlation with hallucination labels "
                "on the validation set. Assigning zero rule weights (gate remains uninformative).",
                stacklevel=2,
            )

        # Recompute G with learned weights
        w_vec = np.array([w_i[rn] for rn in default_rules])
        scores = R @ w_vec  # shape (N,)
        scores = np.clip(scores, 0.0, 1.0)
    else:
        # Fallback if no rule_activations provided
        w_i = {rn: 1.0 / len(default_rules) for rn in default_rules}
        scores = np.array(gate_scores)

    # ── Grid search over threshold pairs ──
    best_score = -1.0
    best_tl = 0.3
    best_th = 0.7

    candidates = np.linspace(0.0, 1.0, 21)

    for tl in candidates:
        for th in candidates:
            if th <= tl:
                continue

            # Compute verdicts
            verdicts = []
            for s in scores:
                if s < tl:
                    verdicts.append(0)  # LOW_RISK → predict not hallucinated
                elif s > th:
                    verdicts.append(1)  # HIGH_RISK → predict hallucinated
                else:
                    verdicts.append(-1)  # AMBIGUOUS

            # Escalation rate
            n_ambiguous = sum(1 for v in verdicts if v == -1)
            esc_rate = n_ambiguous / len(verdicts) if verdicts else 0

            # Skip if escalation rate deviates too much
            if abs(esc_rate - target_escalation_rate) > 0.2:
                continue

            # For non-ambiguous samples, compute AUPRC
            decided_idx = [i for i, v in enumerate(verdicts) if v != -1]
            if len(decided_idx) < 4 or len(set(labels_arr[decided_idx])) < 2:
                continue

            decided_scores = scores[decided_idx]
            decided_labels = labels_arr[decided_idx]
            try:
                ap = average_precision_score(decided_labels, decided_scores)
            except Exception:
                continue

            if ap > best_score:
                best_score = ap
                best_tl = float(tl)
                best_th = float(th)

    # A-6 fix: warn if grid search found no valid candidate
    if best_score < 0:
        warnings.warn(
            "Calibration grid search found no valid threshold pair. "
            "Falling back to defaults (T_L=0.3, T_H=0.7). "
            "Check that validation data has sufficient score variance.",
            stacklevel=2,
        )

    # Ensure valid bounds
    if best_tl >= best_th:
        best_th = min(best_tl + 0.05, 1.0)

    return best_tl, best_th, w_i
