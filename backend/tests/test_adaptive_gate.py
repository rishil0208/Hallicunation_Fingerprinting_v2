"""Tests for adaptive per-model gate — M6."""
from __future__ import annotations

import pytest
import numpy as np

from backend.app.gate.calibrate import calibrate_per_model
from backend.app.gate.score import calibrate_global_thresholds


class TestAdaptiveCalibration:
    def test_per_model_produces_valid_thresholds(self):
        rng = np.random.RandomState(42)
        scores = list(rng.uniform(0, 1, 100))
        labels = [1 if s > 0.5 else 0 for s in scores]
        t_low, t_high, w_i = calibrate_per_model(scores, labels)
        assert 0.0 <= t_low < t_high <= 1.0

    def test_per_model_differs_from_global(self):
        """Core hypothesis: per-model thresholds should differ from global."""
        rng = np.random.RandomState(42)

        # Model A: scores cluster low (mostly correct)
        scores_a = list(rng.uniform(0.0, 0.4, 50)) + list(rng.uniform(0.6, 1.0, 50))
        labels_a = [0] * 50 + [1] * 50

        # Model B: scores cluster high (more hallucinations)
        scores_b = list(rng.uniform(0.0, 0.3, 30)) + list(rng.uniform(0.4, 1.0, 70))
        labels_b = [0] * 30 + [1] * 70

        tl_a, th_a, _ = calibrate_per_model(scores_a, labels_a)
        tl_b, th_b, _ = calibrate_per_model(scores_b, labels_b)

        # Global: pooled
        all_scores = scores_a + scores_b
        all_labels = labels_a + labels_b
        tl_g, th_g, _ = calibrate_global_thresholds(all_scores, all_labels)

        # At least one per-model threshold should differ from global
        # (this is the fundamental thesis of the adaptive gate)
        thresholds_differ = (
            abs(tl_a - tl_g) > 0.01 or abs(th_a - th_g) > 0.01 or
            abs(tl_b - tl_g) > 0.01 or abs(th_b - th_g) > 0.01
        )
        assert thresholds_differ, (
            f"Per-model thresholds should differ from global. "
            f"A=({tl_a:.3f},{th_a:.3f}), B=({tl_b:.3f},{th_b:.3f}), "
            f"G=({tl_g:.3f},{th_g:.3f})"
        )

    def test_per_model_bounds_constrained(self):
        """ADR A-M2: 0 ≤ T_L < T_H ≤ 1."""
        # Edge case: all scores identical
        scores = [0.5] * 50
        labels = [0] * 25 + [1] * 25
        t_low, t_high, _ = calibrate_per_model(scores, labels)
        assert t_low < t_high

    def test_empty_input_returns_defaults(self):
        t_low, t_high, w_i = calibrate_per_model([], [])
        assert t_low == 0.3
        assert t_high == 0.7

    def test_weights_sum_to_one(self):
        scores = list(np.random.uniform(0, 1, 100))
        labels = [1 if s > 0.5 else 0 for s in scores]
        _, _, w_i = calibrate_per_model(
            scores, labels,
            rule_names=["anomaly_pattern_1", "anomaly_pattern_2", "anomaly_pattern_3"],
        )
        assert abs(sum(w_i.values()) - 1.0) < 1e-10
