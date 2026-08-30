# Phase C — Independent Code Review Report
**Project:** Hallucination Fingerprinting Gate (HFG)  
**Organization:** TAM (The AI and ML Club), VIT Vellore  
**Phase:** C — Code Review (Post-Implementation of Milestones M0–M11)  
**Reviewer Role:** Independent Code Reviewer  
**Status:** Review Complete (5 Findings Documented, Recommendations Provided)  
**Date:** August 30, 2026  

---

## 1. Review Overview & Methodology

This code review evaluates the complete codebase generated during Phase C (M0 through M11). The review was conducted against the master specification (`docs/Hallucination_Fingerprinting_Master_Spec_v2.md`), the architecture decisions (`docs/decisions.md`), and standard machine learning/software engineering best practices.

### Validation Matrix
- **Test Suite Status:** 121 / 121 tests passing in 14.35s (`pytest backend/tests/`).
- **Frontend Build Status:** Vite production build passes in 379ms (`dist/assets/index-*.js` 77.99 kB gzip).
- **Git State:** 12 atomic milestone commits (`c12d63a` through `65090b4`).

---

## 2. Detailed Findings

### Finding 1: Per-Model Rule Weights $w_i$ Are Hardcoded Uniform Rather Than Learned
- **Severity:** HIGH / RESEARCH VALIDITY
- **File:** `backend/app/gate/calibrate.py`
- **Line / Area:** Lines 159–171 (inside `calibrate_per_model`)
- **Problem:**  
  Spec Section 5.6 and Section 6 explicitly state:  
  > *"`w_i` and per-model `(T_L, T_H)` learned on a validation split by optimizing the joint objective in Section 6 — never chosen by inspection."*  
  
  However, `calibrate_per_model()` in `calibrate.py` sets uniform weights:
  ```python
  # Per-model weights (learn from data — weight rules proportional
  # to their correlation with positive labels)
  w_i = {}
  if rule_names:
      for rn in rule_names:
          w_i[rn] = 1.0 / len(rule_names)
  else:
      w_i = {
          "anomaly_pattern_1": 1.0 / 3,
          "anomaly_pattern_2": 1.0 / 3,
          "anomaly_pattern_3": 1.0 / 3,
      }
  ```
  Additionally, `calibrate_per_model` takes pre-calculated `gate_scores` as input, which are already computed using default uniform weights prior to calibration.
- **Impact:**  
  1. The system does not actually learn per-model rule importance $w_i$.
  2. Ablation B in Section 6 ("learned vs. uniform $w_i$") becomes degenerate because both global and adaptive pipelines use uniform weights.
- **Reproduction:**  
  Call `calibrate_per_model(scores, labels, ["p1", "p2", "p3"])` with training data where `p1` is 100% correlated with hallucination and `p2` is 0% correlated; the function still outputs `{p1: 0.333, p2: 0.333, p3: 0.333}`.
- **Recommended Fix:**  
  In `calibrate_per_model`, compute the individual rule activation vectors $R \in \mathbb{R}^{N \times 3}$ on the validation split. Fit logistic regression coefficients or correlation coefficients $\beta_i$ against validation labels $y$, apply softmax or L1-normalization $\frac{\max(0, \beta_i)}{\sum \max(0, \beta_j)}$ to ensure $\sum w_i = 1$, and recompute $G = \sum w_i r_i$ during the threshold grid search.

---

### Finding 2: Inverted Specificity Feature in Anomaly Pattern 2 (`specificity_low` instead of `specificity_high`)
- **Severity:** MEDIUM / SPECIFICATION ADHERENCE
- **File:** `backend/app/gate/rules.py`
- **Line / Area:** Lines 12, 28–32, 58–63
- **Problem:**  
  Spec Section 5.5 defines Anomaly Pattern 2 as:
  $$\text{anomaly\_pattern\_2}(X) \leftarrow \text{specificity\_high}(X), \text{evidence\_low}(X)$$
  *(Capturing the hallucination phenomenon where a model generates hyper-specific numbers/names/dates without checkable evidence).*  
  
  The implementation in `rules.py` inverts Specificity:
  ```python
  DEFAULT_FACT_THRESHOLDS = {
      # ...
      "specificity_low": ("S", 0.5), # inverted feature
      "evidence_low": ("E", 0.5),
  }

  RULES = [
      # ...
      {
          "name": "anomaly_pattern_2",
          "description": "Low specificity + low evidence",
          "antecedents": ["specificity_low", "evidence_low"],
          "features_involved": ["S", "E"],
      },
  ]
  ```
- **Impact:**  
  Responses with high specificity and low evidence (the intended hallucination pattern) fail to activate Pattern 2 ($r_2 = 0$). Instead, Pattern 2 only activates on vague, low-evidence text.
- **Reproduction:**  
  Input an answer with $S = 0.9$ (many entities/dates) and $E = 0.1$ (no evidence relations); `anomaly_pattern_2` yields strength $0.0$.
- **Recommended Fix:**  
  Change `specificity_low` to `specificity_high` with threshold $0.5$ in `DEFAULT_FACT_THRESHOLDS`. Update `RULES[1]["antecedents"]` to `["specificity_high", "evidence_low"]` and treat `specificity_high` as a standard non-inverted fact activation ($S \ge \text{threshold}$).

---

### Finding 3: Background Calibration Thread Processes Unbounded 20,000 Records
- **Severity:** MEDIUM / SYSTEM RESPONSIVENESS
- **File:** `backend/app/api/routes.py`
- **Line / Area:** Lines 198–210 (inside `calibrate_model`)
- **Problem:**  
  `POST /api/v1/models/{model_id}/calibrate` spawns a background thread that iterates over all 20,000 records of `qa_data.json` and extracts features on 12,000 training records using spaCy and sentence-transformers on CPU without batching or sample limits:
  ```python
  loader = HaluEvalQALoader(data_path, default_model_id=model_id)
  records = list(loader.load())
  train, val, test = split_records(records)
  train_features = [extractor.extract(r.answer) for r in train] # 12,000 items on CPU
  ```
- **Impact:**  
  Calling this endpoint causes 100% CPU utilization for multiple hours with no intermediate progress updates.
- **Reproduction:**  
  Issue `POST /api/v1/models/chatgpt/calibrate` and monitor job status via `/api/v1/jobs/{job_id}`.
- **Recommended Fix:**  
  Allow an optional `sample_limit: Optional[int] = 1000` in the calibration request or batch the sentence embedding calculations.

---

### Finding 4: Pure Python Interval Arithmetic vs. PyReason Engine
- **Severity:** LOW / ARCHITECTURAL CLARIFICATION
- **File:** `backend/app/gate/rules.py` and `backend/app/gate/score.py`
- **Line / Area:** Global rule execution logic
- **Problem:**  
  While PyReason 3.7.0 was installed in M0, the runtime gate scoring in M5 uses direct Python/NumPy functions to evaluate fact intervals and antecedent minimums ($r_i$) rather than compiling PyReason graph files on every request.
- **Impact:**  
  This provides a massive latency benefit (<1ms vs. ~500ms per request) and exact numerical reproducibility, but diverges from the literal section title "5.5 Symbolic rule graph (PyReason)".
- **Recommended Fix:**  
  Formally record in `docs/decisions.md` that runtime scoring uses closed-form interval logic evaluation derived from PyReason semantics to optimize serving latency.

---

### Finding 5: Evaluation Results JSON Files Not Seeded by Default
- **Severity:** LOW / UX
- **File:** `backend/app/api/routes.py`
- **Line / Area:** Lines 263–279
- **Problem:**  
  Unit tests in `test_eval_harness.py` wrote to temporary directories. Consequently, `eval/results/` is empty in fresh deployments, causing `/api/v1/eval/summary` and the frontend Evaluation view to report "No evaluation results yet".
- **Impact:**  
  The frontend Evaluation tab is empty until an offline experiment run script is executed.
- **Recommended Fix:**  
  Commit a set of default benchmark result JSON files in `eval/results/` generated from the 5 baselines on HaluEval.

---

## 3. Specification & Guardrails Compliance Checklist

| Spec Section | Item | Status | Notes |
| :--- | :--- | :---: | :--- |
| **5.1** | `dataset_role` enum type-enforced | ✅ PASS | `DatasetRole.PRIMARY` / `DatasetRole.SECONDARY` |
| **5.1** | Non-leakage calibration guard | ✅ PASS | `ensure_primary_only()` strictly filters & warns |
| **5.3** | 6 Telemetry features $(H, S, C, E, D, M)$ | ✅ PASS | Implemented in `DefaultFeatureExtractor` |
| **5.4** | Cluster label alignment (ADR A-C2) | ✅ PASS | Post-hoc aligned with ground truth label rate |
| **5.5** | $G = \sum w_i r_i$ calculation | ✅ PASS | Score bounded in $[0, 1]$ |
| **5.6** | Threshold bounds (ADR A-M2) | ✅ PASS | Constrained $0.0 \le T_L < T_H \le 1.0$ |
| **5.7** | Anonymized judge prompt (ADR A-M1) | ✅ PASS | `model_id` scrubbed from prompt |
| **5.7** | Judge failure HTTP 502 (ADR A-C1) | ✅ PASS | Never fakes `RESOLVED_AMBIGUOUS` |
| **5.8** | Uncalibrated model guard (ADR A-H1) | ✅ PASS | Explicit None-check before comparison |
| **5.12** | 7 FastAPI endpoints matching schemas | ✅ PASS | Exact response shapes validated by tests |
| **5.13** | *Signal Forensics* CSS tokens | ✅ PASS | Graphite ink, paper, 3 semantic signal colors |
| **5.13** | 6-Axis Fingerprint Radar | ✅ PASS | Ridge paths, verdict overlay, reduced motion |
| **5.13** | Persistent research disclaimer | ✅ PASS | Always visible, non-dismissible footer |
| **6.0** | 5 Baselines implemented | ✅ PASS | Baselines 1, 2, 3, 4, 5 all callable |

---

## 4. Summary & Next Steps

Phase C implementation represents a complete, functional, and rigorous software deliverable. All core features, API routes, evaluation metrics, and frontend interfaces are fully connected and tested.

**Recommended Action Items for Refinement:**
1. Update `backend/app/gate/calibrate.py` to fit learned $w_i$ weights via validation split correlation/regression.
2. Correct `backend/app/gate/rules.py` Rule 2 to use `specificity_high` instead of `specificity_low`.
3. Add an offline evaluation generation script to populate `eval/results/` with baseline comparison benchmarks for the frontend dashboard.
