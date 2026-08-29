"""Per-model fingerprint clustering with label-aligned k-means."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from backend.app.schemas import (
    ClusterInfo,
    DatasetRole,
    Fingerprint,
    NormalizationParams,
    Record,
)

FEATURE_KEYS = ["H", "S", "C", "E", "D", "M"]


def compute_normalization(
    feature_vectors: list[dict[str, float]],
) -> NormalizationParams:
    """Compute per-feature min/max from a list of feature dicts."""
    if not feature_vectors:
        return NormalizationParams(
            feature_mins={k: 0.0 for k in FEATURE_KEYS},
            feature_maxs={k: 1.0 for k in FEATURE_KEYS},
        )

    mins = {k: float("inf") for k in FEATURE_KEYS}
    maxs = {k: float("-inf") for k in FEATURE_KEYS}

    for fv in feature_vectors:
        for k in FEATURE_KEYS:
            val = fv.get(k, 0.0)
            if val < mins[k]:
                mins[k] = val
            if val > maxs[k]:
                maxs[k] = val

    # Avoid zero-range features
    for k in FEATURE_KEYS:
        if mins[k] == maxs[k]:
            maxs[k] = mins[k] + 1.0

    return NormalizationParams(feature_mins=mins, feature_maxs=maxs)


def normalize_features(
    features: dict[str, float],
    params: NormalizationParams,
) -> dict[str, float]:
    """Normalize features to [0,1] using pre-computed min/max."""
    normalized = {}
    for k in FEATURE_KEYS:
        val = features.get(k, 0.0)
        lo = params.feature_mins[k]
        hi = params.feature_maxs[k]
        if hi == lo:
            normalized[k] = 0.0
        else:
            normalized[k] = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    return normalized


def _features_to_matrix(feature_vectors: list[dict[str, float]]) -> np.ndarray:
    """Convert list of feature dicts to numpy array."""
    return np.array([[fv.get(k, 0.0) for k in FEATURE_KEYS] for fv in feature_vectors])


def cluster_fingerprint(
    feature_vectors: list[dict[str, float]],
    labels: list[int],
    k: int = 2,
    seed: int = 42,
) -> ClusterInfo:
    """Run k-means and post-hoc align clusters with ground-truth labels.

    Per ADR A-C2: k-means assigns arbitrary cluster IDs. We align them
    by computing mean label per cluster and assigning 'hallucination_region'
    to the cluster with the higher mean hallucination rate.
    """
    X = _features_to_matrix(feature_vectors)
    labels_arr = np.array(labels)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    # Post-hoc label alignment
    cluster_labels = []
    for c in range(k):
        mask = cluster_ids == c
        if mask.sum() == 0:
            cluster_labels.append(f"cluster_{c}")
            continue
        mean_label = labels_arr[mask].mean()
        if mean_label >= 0.5:
            cluster_labels.append("hallucination_region")
        else:
            cluster_labels.append("correct_region")

    # Handle tie: if all clusters got the same label, differentiate by rate
    if len(set(cluster_labels)) == 1 and k >= 2:
        rates = []
        for c in range(k):
            mask = cluster_ids == c
            rates.append(labels_arr[mask].mean() if mask.sum() > 0 else 0.0)
        best = int(np.argmax(rates))
        cluster_labels = [
            "hallucination_region" if c == best else "correct_region"
            for c in range(k)
        ]

    centroids = []
    for c in range(k):
        centroid_dict = {
            FEATURE_KEYS[i]: float(kmeans.cluster_centers_[c, i])
            for i in range(len(FEATURE_KEYS))
        }
        centroids.append(centroid_dict)

    return ClusterInfo(centroids=centroids, cluster_labels=cluster_labels)


def build_fingerprint(
    model_id: str,
    feature_vectors: list[dict[str, float]],
    labels: list[int],
    k: int = 2,
    seed: int = 42,
    version: str = "v1",
) -> Fingerprint:
    """Build a complete fingerprint for a model.

    1. Compute normalization params from raw features
    2. Normalize all features
    3. Cluster normalized features with label alignment
    """
    norm_params = compute_normalization(feature_vectors)

    normalized = [normalize_features(fv, norm_params) for fv in feature_vectors]

    clusters = cluster_fingerprint(normalized, labels, k=k, seed=seed)

    return Fingerprint(
        model_id=model_id,
        version=version,
        normalization=norm_params,
        clusters=clusters,
        calibration_dataset_size=len(feature_vectors),
    )
