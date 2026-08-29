from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DatasetRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class Record(BaseModel):
    question: str
    answer: str
    model_id: str
    label: int = Field(ge=0, le=1)  # 0 = correct, 1 = hallucinated
    source_dataset: str
    dataset_role: DatasetRole


class NormalizationParams(BaseModel):
    feature_mins: dict[str, float]
    feature_maxs: dict[str, float]


class ClusterInfo(BaseModel):
    centroids: list[dict[str, float]]
    cluster_labels: list[str]


class Fingerprint(BaseModel):
    model_id: str
    version: str
    normalization: NormalizationParams
    clusters: Optional[ClusterInfo] = None
    w_i: Optional[dict[str, float]] = None
    t_low: Optional[float] = None
    t_high: Optional[float] = None
    calibration_dataset_size: int = 0
    last_calibrated_at: str = ""


class TriggeredPattern(BaseModel):
    name: str
    features_involved: list[str]
    strength: float


class Thresholds(BaseModel):
    t_low: float
    t_high: float


class GateEvaluation(BaseModel):
    gate_score: float
    gate_verdict: str
    thresholds: Thresholds
    triggered_patterns: list[TriggeredPattern]
    feature_breakdown: dict[str, float]
    raw_features: dict[str, float]
    model_id: str
    fingerprint_version: str


class JudgeResult(BaseModel):
    verdict: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class GateResult(BaseModel):
    verdict: str
    gate_score: float
    thresholds: Thresholds
    resolved_by: str
    triggered_patterns: list[TriggeredPattern]
    feature_breakdown: dict[str, float]
    explanation: str
    explanation_source: str
    model_id: str
    fingerprint_version: str
