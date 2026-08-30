# Phase C — Adversarial Test Review Report
**Project:** Hallucination Fingerprinting Gate (HFG)  
**Organization:** TAM (The AI and ML Club), VIT Vellore  
**Phase:** C — Adversarial Test Review (Post-Implementation)  
**Reviewer Role:** Adversarial Breaker  
**Status:** Review Complete (6 Vulnerabilities & Edge-Case Failure Modes Identified)  
**Date:** August 30, 2026  

---

## 1. Executive Summary

An adversarial inspection was conducted against the complete Phase C codebase (`backend/app/`, `eval/`, `frontend/`, and `routes.py`). The objective was to challenge the implementation across memory safety, concurrent request handling, LLM prompt parsing, threshold boundary math, and numerical stability.

A total of **6 concrete failure modes and edge-case vulnerabilities** were identified.

---

## 2. Adversarial Findings & Vulnerability Details

### Finding A-1: Unbounded Request Payload Memory Exhaustion (DoS via spaCy Tokenization)
- **Severity:** HIGH / SECURITY & DOS
- **Target Component:** `backend/app/plugins/features/default.py` (`DefaultFeatureExtractor`) & `backend/app/api/routes.py` (`ScoreRequest`)
- **Vulnerability Mechanism:**  
  `ScoreRequest` defines `answer: str` without maximum length validation. When passed to `DefaultFeatureExtractor.extract()`, the raw text is passed directly into spaCy's sentence parser `_get_nlp()(answer)`.
- **Adversarial Input / Reproduction:**  
  Send a payload containing a 10MB text string to `POST /api/v1/score`:
  ```json
  {
    "answer": "word ".repeat(2000000),
    "model_id": "chatgpt"
  }
  ```
- **Impact:**  
  spaCy attempts to construct an in-memory document with 2,000,000 token objects on the worker thread, blocking FastAPI threadpool workers, leading to high CPU starvation, memory spikes (`MemoryError`), and potential server crash.
- **Breaker Verdict:** **SYSTEM CRASH / DOS VULNERABILITY**

---

### Finding A-2: Race Condition & Resource Starvation in Background Calibration
- **Severity:** HIGH / CONCURRENCY & STABILITY
- **Target Component:** `backend/app/api/routes.py` (lines 182–248, `calibrate_model`)
- **Vulnerability Mechanism:**  
  `calibrate_model` spawns an un-throttled background daemon `Thread(target=_run_calibration)` that writes to `_fingerprints[model_id]` and `_jobs[job_id]` directly without a thread lock or concurrency limit.
- **Adversarial Input / Reproduction:**  
  Issue 10 concurrent `POST /api/v1/models/chatgpt/calibrate` HTTP requests simultaneously.
- **Impact:**  
  10 concurrent background threads will spawn, each processing 12,000 training records via spaCy + sentence-transformers simultaneously. CPU utilization pegs at 100%, thread contexts lock up, and non-atomic dictionary mutations cause race conditions where whichever thread finishes last arbitrarily overwrites the global `_fingerprints[model_id]` store.
- **Breaker Verdict:** **CONCURRENCY RACE CONDITION**

---

### Finding A-3: LLM Judge Failure on Conversational Prefix before Code Fences
- **Severity:** MEDIUM / PARSING BRICKING
- **Target Component:** `backend/app/plugins/judges/gemini_judge.py` (lines 117–123)
- **Vulnerability Mechanism:**  
  `GeminiJudgePlugin.judge()` attempts to strip markdown code fences using a naive prefix check:
  ```python
  if text.startswith("```"):
      text = text.split("```")[1]
      if text.startswith("json"):
          text = text[4:]
      text = text.strip()
  ```
- **Adversarial Input / Reproduction:**  
  If Gemini outputs a standard introductory phrase before the JSON fence:
  ```
  Here is my analysis:
  ```json
  {
    "verdict": "HIGH_RISK",
    "explanation": "Multiple markers found.",
    "confidence": 0.85
  }
  ```
  ```
- **Impact:**  
  `text.startswith("```")` evaluates to `False`. The code passes the entire string directly to `json.loads(text)`, raising a `json.JSONDecodeError`. The catch block catches this and raises `GeminiJudgeError`, forcing `routes.py` to return an **HTTP 502 Bad Gateway** error to the end user even though a perfectly valid JSON payload was returned by the LLM.
- **Breaker Verdict:** **FALSE-POSITIVE 502 FAILURE**

---

### Finding A-4: Degenerate Threshold Lock when $T_L = 1.0$
- **Severity:** MEDIUM / BOUNDARY CORRECTION BUGS
- **Target Component:** `backend/app/gate/score.py` (lines 113–116, `classify`)
- **Vulnerability Mechanism:**  
  To satisfy ADR A-M2 ($0.0 \le T_L < T_H \le 1.0$), `classify()` enforces:
  ```python
  t_low = max(0.0, min(t_low, 1.0))
  t_high = max(0.0, min(t_high, 1.0))
  if t_low >= t_high:
      t_high = min(t_low + 0.05, 1.0)
  ```
- **Adversarial Input / Reproduction:**  
  Calibrate a model where validation scores skew extreme high, yielding $T_L = 1.0$.
- **Impact:**  
  `t_low` becomes `1.0`. `t_low + 0.05` equals `1.05`. `min(1.05, 1.0)` clamps `t_high` to `1.0`.  
  Now $T_L = 1.0$ and $T_H = 1.0$.  
  Evaluating any score $G$:
  - If $G < 1.0 \rightarrow \text{LOW\_RISK}$
  - If $G = 1.0 \rightarrow \text{AMBIGUOUS}$ (since $1.0 \le G \le 1.0$)
  - $G > 1.0$ is mathematically impossible since $G \in [0, 1]$.
  Thus, **`HIGH_RISK` verdict becomes completely unreachable** for this model.
- **Breaker Verdict:** **UNREACHABLE STATE BUG**

---

### Finding A-5: Evaluation Pipeline Crash on `NaN` Feature Outputs
- **Severity:** MEDIUM / NUMERICAL INSTABILITY
- **Target Component:** `eval/metrics.py` (lines 10–13, `auroc`)
- **Vulnerability Mechanism:**  
  `auroc()` calls `roc_auc_score(y_true, y_score)` directly without scrubbing `NaN` values:
  ```python
  def auroc(y_true: list[int], y_score: list[float]) -> float:
      if len(set(y_true)) < 2:
          return float("nan")
      return float(roc_auc_score(y_true, y_score))
  ```
- **Adversarial Input / Reproduction:**  
  If any raw prediction score $p["score"]$ contains `NaN` (e.g. from an empty document or feature extractor edge case), `roc_auc_score` raises `ValueError: Input contains NaN, infinity or a value too large for dtype('float64')`.
- **Impact:**  
  The entire evaluation experiment run crashes mid-benchmark.
- **Breaker Verdict:** **BENCHMARK PIPELINE CRASH**

---

### Finding A-6: Silent Fallback to Default Thresholds on Uniform Scores
- **Severity:** LOW / CALIBRATION SILENT FAILURES
- **Target Component:** `backend/app/gate/calibrate.py` (lines 139–141)
- **Vulnerability Mechanism:**  
  During per-model calibration grid search over validation scores:
  ```python
  decided_idx = [i for i, v in enumerate(verdicts) if v != -1]
  if len(decided_idx) < 4 or len(set(labels_arr[decided_idx])) < 2:
      continue
  ```
- **Adversarial Input / Reproduction:**  
  Pass a validation set where all answers yield constant features (e.g. synthetic test set or short template answers where $G = 0.0$ for all records).
- **Impact:**  
  Every candidate threshold pair fails the `len(set(labels_arr[decided_idx])) < 2` guard. The grid search terminates with `best_score = -1.0` and silently falls back to `t_low = 0.3, t_high = 0.7` without logging any warning or flagging an uncalibrated status to the user.
- **Breaker Verdict:** **SILENT CALIBRATION FALLBACK**

---

## 3. Adversarial Findings Summary Matrix

| Finding | Target Component | Root Cause | Failure Mode & Impact |
| :--- | :--- | :--- | :--- |
| **A-1** | `DefaultFeatureExtractor` | Unbounded input string passed to spaCy parser | Server memory crash / DoS |
| **A-2** | `routes.py` (`/calibrate`) | Un-throttled daemon threads mutating global dict | Concurrency race condition |
| **A-3** | `GeminiJudgePlugin` | Prefix-only markdown fence matching | Valid LLM JSON returns HTTP 502 |
| **A-4** | `score.py` (`classify`) | `min(t_low + 0.05, 1.0)` when $T_L = 1.0$ | `HIGH_RISK` verdict unreachable |
| **A-5** | `eval/metrics.py` | Unsanitized `NaN` passed to `scikit-learn` ROC | Benchmark harness crashes |
| **A-6** | `calibrate.py` | Failed grid search falls back silently | Unnoticed default threshold fallback |
