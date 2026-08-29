# Architecture Decision Records

Decisions made during implementation, recorded as they arise.

---

## ADR-1: Extracting r_i from PyReason Bounds

**Status**: Accepted  
**Context**: PyReason returns `[lower, upper]` interval bounds for each rule conclusion.
The spec defines `G = Σ w_i * r_i` but doesn't specify how `r_i` is derived.  
**Decision**: Default `r_i = lower` (conservative). Configurable via `gate.r_i_strategy`
in YAML (`"lower"`, `"upper"`, `"midpoint"`).  

## ADR-2: Train/Validation/Test Split

**Status**: Accepted  
**Context**: Thresholds and weights are learned on validation data.  
**Decision**: 60/20/20 stratified by label, per model, seed=42. Indices persisted in
`data/processed/splits/{model_id}_split.json`.

## ADR-3: Fingerprint Serialization

**Status**: Accepted  
**Decision**: JSON at `data/processed/fingerprints/{model_id}_v{version}.json`,
validated by `Fingerprint` Pydantic model.

## ADR-4: Per-Model Normalization Storage

**Status**: Accepted  
**Decision**: `NormalizationParams` stored inside `Fingerprint` object.

## ADR-5: RESOLVED_AMBIGUOUS Semantics

**Status**: Accepted  
**Decision**: Accept the spec as-is. API returns `RESOLVED_AMBIGUOUS`; no `resolved_direction`
field is added beyond the spec.

## ADR-6: Plugin Selection Mechanism

**Status**: Accepted  
**Decision**: YAML config + simple dict-based registry. Spec says "do not reach for a
heavyweight plugin-discovery framework."

## ADR-7: Cluster-Label Alignment

**Status**: Accepted  
**Decision**: After k-means, compute mean hallucination label rate per cluster. Assign
`"hallucination_region"` to the cluster with higher mean hallucination rate.

## ADR-8: Judge Failure Behavior

**Status**: Accepted  
**Decision**: On judge failure, return HTTP 502 with error body containing `gate_score` and
`gate_verdict: "AMBIGUOUS"`. Never return `resolved_by: "llm_judge"` when the judge errored.
