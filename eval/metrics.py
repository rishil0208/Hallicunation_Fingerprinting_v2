"""Evaluation metrics — AUROC, AUPRC, ECE, escalation rate."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(y_true: list[int], y_score: list[float]) -> float:
    """Area Under ROC Curve."""
    if len(set(y_true)) < 2:
        return float("nan")
    # A-5 fix: scrub NaN/inf values
    clean = [(t, s) for t, s in zip(y_true, y_score) if np.isfinite(s)]
    if len(clean) < 2 or len(set(t for t, _ in clean)) < 2:
        return float("nan")
    ct, cs = zip(*clean)
    return float(roc_auc_score(list(ct), list(cs)))


def auprc(y_true: list[int], y_score: list[float]) -> float:
    """Area Under Precision-Recall Curve (primary metric per spec)."""
    if len(set(y_true)) < 2:
        return float("nan")
    # A-5 fix: scrub NaN/inf values
    clean = [(t, s) for t, s in zip(y_true, y_score) if np.isfinite(s)]
    if len(clean) < 2 or len(set(t for t, _ in clean)) < 2:
        return float("nan")
    ct, cs = zip(*clean)
    return float(average_precision_score(list(ct), list(cs)))


def ece(y_true: list[int], y_score: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Required per spec if G is ever presented as a probability.
    Bins predictions into n_bins equal-width bins, computes
    |accuracy - confidence| weighted by bin size.
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_score_arr = np.array(y_score, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true_arr)
    if total == 0:
        return 0.0

    calibration_error = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_score_arr >= lo) & (y_score_arr <= hi)
        else:
            mask = (y_score_arr >= lo) & (y_score_arr < hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_acc = y_true_arr[mask].mean()
        bin_conf = y_score_arr[mask].mean()
        calibration_error += (bin_size / total) * abs(bin_acc - bin_conf)

    return float(calibration_error)


def escalation_rate(verdicts: list[str]) -> float:
    """Fraction of verdicts that required LLM judge escalation."""
    if not verdicts:
        return 0.0
    escalated = sum(1 for v in verdicts if v == "RESOLVED_AMBIGUOUS")
    return escalated / len(verdicts)
