# SKEPTICAL ARCHITECTURE AUDIT — Hallucination Fingerprinting System

**Date**: 2026-08-29  
**Phase**: B — Independent Architecture Review  
**Role**: Skeptical Senior ML/Research Engineer  
**Repository**: `/home/rishil-n/AI_PROJECT`

---

## CRITICAL FINDINGS

1. **SEVERITY: CRITICAL / COMPONENT: Gate & Judge Pipeline (`score.py`) / PROBLEM: Tight coupling of Stage 2 G score calculation and Stage 3 Judge Escalation / WHY IT MATTERS: Prevents running Baseline 2 (never-escalate/gate-only) in the evaluation harness without duplicate code, and prevents isolating G score computation for metrics like AUROC and ECE across the entire test set. / RECOMMENDATION: Refactor `score_answer` to call a separate function `compute_gate_score(answer, model_id) -> GateScoreResult` (returning G, verdict, triggered patterns) and then conditionally call `escalate_to_judge` if verdict is AMBIGUOUS and escalation is enabled.**
2. **SEVERITY: CRITICAL / COMPONENT: Environment (M0) / PROBLEM: Unresolved system package dependencies (missing pip and Python 3.14 compatibility) / WHY IT MATTERS: The project cannot build or run without a way to install packages. Python 3.14 is a pre-release version that has no wheels for JPype1 or heavy ML packages like scikit-learn, meaning pip installs will fail on compilation. / RECOMMENDATION: Prioritize setting up a Python 3.10 virtual environment using apt-get for python3.10-venv and python3.10-dev.**

---

## HIGH-PRIORITY FINDINGS

1. **SEVERITY: HIGH / COMPONENT: Evaluation & Baselines (M8 / M5) / PROBLEM: Research confounders in Baseline 3 (Global Threshold) comparison / WHY IT MATTERS: If the global baseline uses per-model feature normalizations or is compared against an adaptive gate that learns weights $w_i$ while the global gate uses uniform weights, the hypothesis of "adaptive thresholds" is not isolated from "weight learning" and "feature normalization". / RECOMMENDATION: Ensure Baseline 3 has two variants: (a) a true ablation with uniform weights and global normalization, and (b) a variant with global-only learned weights and global-only normalization. Clearly document this split in the evaluation summary.**
2. **SEVERITY: HIGH / COMPONENT: Ingestion & Calibration (M1 / M6) / PROBLEM: No type-level enforcement of dataset role in calibration / WHY IT MATTERS: While the `Record` schema has `dataset_role`, the calibration module accepts a generic iterable of records and relies on convention to filter. A bug in the filtering code could silently pool secondary (TruthfulQA) records into calibration, causing evaluation leakage. / RECOMMENDATION: Define a `PrimaryRecord` subclass or use a wrapper `CalibrationDataset` type that explicitly enforces `dataset_role == DatasetRole.PRIMARY` on initialization, or add a strict runtime assertion in `calibrate.py` that raises a ValueError if any record with `DatasetRole.SECONDARY` is passed.**

---

## MEDIUM FINDINGS

1. **SEVERITY: MEDIUM / COMPONENT: Judge Escalation (M7) / PROBLEM: Model identity leakage to LLM Judge / WHY IT MATTERS: If the judge is given the `model_id` or the model name (e.g. "chatgpt"), it can use its own pre-trained priors about the model's accuracy rather than evaluating the response's stylistic telemetry. This inflates performance metrics artificially. / RECOMMENDATION: Anonymize the model identity in the prompt (e.g. "Model A") and present the fingerprint centroids and thresholds abstractly without any model names.**
2. **SEVERITY: MEDIUM / COMPONENT: Symbolic Gate (M5) / PROBLEM: Lack of concrete conversion from PyReason bounds to rule activation strength `r_i` / WHY IT MATTERS: The spec defines `G = Σ w_i * r_i` but does not specify how `r_i` is derived from the PyReason interval `[lower, upper]`. The choice of `r_i = lower` is safe but could be overly conservative, leading to high escalation rates. / RECOMMENDATION: Implement a configurable conversion strategy in `configs/default.yaml` allowing easy switching between `lower`, `upper`, and `midpoint` to test their impact during M8 ablations.**
3. **SEVERITY: MEDIUM / COMPONENT: API Contract (M9) / PROBLEM: Lack of structured resolution direction in RESOLVED_AMBIGUOUS verdict / WHY IT MATTERS: When the API returns `RESOLVED_AMBIGUOUS`, the client does not get a structured field indicating whether the judge resolved the risk to low or high. Parsing the explanation string is brittle. / RECOMMENDATION: Add a nullable `resolved_verdict: "LOW_RISK" | "HIGH_RISK" | null` to the Pydantic schema of `GateResult` to expose the judge's final verdict in a structured way (this is a minor non-breaking extension).**

---

## MISSED INFORMATION

1. The VS Code backup file at `/home/rishil-n/.config/Code/Backups/...` is a replica of the spec file with VS Code metadata on line 1, not an independent version.
2. The `AltFeatureExtractor` is intended to use alternative lexicons to verify the plugin architecture is modular and configuration-driven.

---

## CORRECTIONS TO DISCOVERY REPORT

1. **PyReason System Dependencies**: Correct the claim that PyReason requires a Java JRE/JDK system-level dependency. PyReason runs on native Python using Numba JIT.
2. **Python Version**: Highlight that Python 3.14 is a pre-release version that presents high installation and compilation risks for compiled dependencies (Numba, JPype, scikit-learn).

---

## QUESTIONS THAT REQUIRE HUMAN DECISION

1. **Anonymization of LLM Judge**: Should the model's identity be anonymized in the LLM Judge prompt to prevent model-specific prior bias?
2. **Global Baseline Calibration**: How should feature normalization and weights be handled for the global baseline to ensure a fair comparison against the per-model adaptive gate?
3. **Structured Resolved Verdict**: Should we add `resolved_verdict` to the API schema so clients can read the judge's binary decision programmatically?
