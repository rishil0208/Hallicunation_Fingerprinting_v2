"""Tests for fingerprint clustering — M4."""
from __future__ import annotations

import pytest
import numpy as np

from backend.app.schemas import DatasetRole, Record
from backend.app.fingerprint.cluster import (
    FEATURE_KEYS,
    build_fingerprint,
    cluster_fingerprint,
    compute_normalization,
    normalize_features,
)
from backend.app.gate.calibrate import ensure_primary_only, split_records


def _make_feature_vectors(n: int = 100, seed: int = 42) -> tuple[list[dict], list[int]]:
    """Generate synthetic feature vectors with known cluster structure."""
    rng = np.random.RandomState(seed)

    features = []
    labels = []

    # Cluster 0: correct answers — low H, high S, low C, high E
    for _ in range(n // 2):
        fv = {
            "H": rng.uniform(0.0, 2.0),
            "S": rng.uniform(0.3, 0.8),
            "C": rng.uniform(0.0, 0.2),
            "E": rng.uniform(0.5, 1.0),
            "D": rng.uniform(0.0, 0.01),
            "M": rng.uniform(0.0, 3.0),
        }
        features.append(fv)
        labels.append(0)

    # Cluster 1: hallucinated answers — high H, low S, high C, low E
    for _ in range(n // 2):
        fv = {
            "H": rng.uniform(5.0, 10.0),
            "S": rng.uniform(0.0, 0.2),
            "C": rng.uniform(0.5, 1.0),
            "E": rng.uniform(0.0, 0.3),
            "D": rng.uniform(0.01, 0.05),
            "M": rng.uniform(5.0, 10.0),
        }
        features.append(fv)
        labels.append(1)

    return features, labels


# ── Normalization tests ──

class TestNormalization:
    def test_normalization_produces_01_range(self):
        features, _ = _make_feature_vectors()
        params = compute_normalization(features)
        for fv in features:
            normalized = normalize_features(fv, params)
            for k in FEATURE_KEYS:
                assert 0.0 <= normalized[k] <= 1.0, f"{k} out of [0,1]: {normalized[k]}"

    def test_normalization_min_maps_to_zero(self):
        features = [
            {"H": 1.0, "S": 0.5, "C": 0.0, "E": 0.0, "D": 0.0, "M": 0.0},
            {"H": 5.0, "S": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "M": 1.0},
        ]
        params = compute_normalization(features)
        # The record with min H should normalize to 0
        result = normalize_features(features[0], params)
        assert result["H"] == 0.0

    def test_normalization_max_maps_to_one(self):
        features = [
            {"H": 1.0, "S": 0.5, "C": 0.0, "E": 0.0, "D": 0.0, "M": 0.0},
            {"H": 5.0, "S": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "M": 1.0},
        ]
        params = compute_normalization(features)
        result = normalize_features(features[1], params)
        assert result["H"] == 1.0

    def test_empty_features_produces_defaults(self):
        params = compute_normalization([])
        assert all(params.feature_mins[k] == 0.0 for k in FEATURE_KEYS)
        assert all(params.feature_maxs[k] == 1.0 for k in FEATURE_KEYS)


# ── Clustering tests ──

class TestClustering:
    def test_cluster_produces_k_clusters(self):
        features, labels = _make_feature_vectors()
        params = compute_normalization(features)
        normalized = [normalize_features(fv, params) for fv in features]
        clusters = cluster_fingerprint(normalized, labels, k=2)
        assert len(clusters.centroids) == 2
        assert len(clusters.cluster_labels) == 2

    def test_cluster_labels_are_aligned(self):
        """Post-hoc alignment: hallucination cluster should have higher mean label."""
        features, labels = _make_feature_vectors(n=200)
        params = compute_normalization(features)
        normalized = [normalize_features(fv, params) for fv in features]
        clusters = cluster_fingerprint(normalized, labels, k=2)
        assert "hallucination_region" in clusters.cluster_labels
        assert "correct_region" in clusters.cluster_labels

    def test_centroids_have_all_features(self):
        features, labels = _make_feature_vectors()
        params = compute_normalization(features)
        normalized = [normalize_features(fv, params) for fv in features]
        clusters = cluster_fingerprint(normalized, labels, k=2)
        for centroid in clusters.centroids:
            assert set(centroid.keys()) == set(FEATURE_KEYS)


# ── Full fingerprint tests ──

class TestBuildFingerprint:
    def test_builds_complete_fingerprint(self):
        features, labels = _make_feature_vectors()
        fp = build_fingerprint("test_model", features, labels)
        assert fp.model_id == "test_model"
        assert fp.version == "v1"
        assert fp.normalization is not None
        assert fp.clusters is not None
        assert fp.calibration_dataset_size == len(features)

    def test_fingerprint_reproducible(self):
        features, labels = _make_feature_vectors()
        fp1 = build_fingerprint("m1", features, labels, seed=42)
        fp2 = build_fingerprint("m1", features, labels, seed=42)
        assert fp1.clusters.centroids == fp2.clusters.centroids


# ── Calibration safety tests ──

class TestCalibrationSafety:
    def test_ensure_primary_only_filters_secondary(self):
        records = [
            Record(question="Q", answer="A", model_id="m", label=0,
                   source_dataset="s", dataset_role=DatasetRole.PRIMARY),
            Record(question="Q", answer="A", model_id="m", label=0,
                   source_dataset="s", dataset_role=DatasetRole.SECONDARY),
        ]
        with pytest.warns(UserWarning, match="Filtered 1"):
            result = ensure_primary_only(records)
        assert len(result) == 1
        assert result[0].dataset_role == DatasetRole.PRIMARY

    def test_ensure_primary_only_raises_on_empty(self):
        records = [
            Record(question="Q", answer="A", model_id="m", label=0,
                   source_dataset="s", dataset_role=DatasetRole.SECONDARY),
        ]
        with pytest.raises(ValueError, match="No primary records"):
            ensure_primary_only(records)

    def test_split_reproducible(self):
        records = [
            Record(question=f"Q{i}", answer=f"A{i}", model_id="m",
                   label=i % 2, source_dataset="s",
                   dataset_role=DatasetRole.PRIMARY)
            for i in range(100)
        ]
        t1, v1, te1 = split_records(records, seed=42)
        t2, v2, te2 = split_records(records, seed=42)
        assert [r.question for r in t1] == [r.question for r in t2]

    def test_split_proportions(self):
        records = [
            Record(question=f"Q{i}", answer=f"A{i}", model_id="m",
                   label=0, source_dataset="s",
                   dataset_role=DatasetRole.PRIMARY)
            for i in range(100)
        ]
        train, val, test = split_records(records)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20
