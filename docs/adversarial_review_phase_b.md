# ADVERSARIAL ARCHITECTURE REVIEW — Hallucination Fingerprinting System

**Date**: 2026-08-29  
**Phase**: B — Adversarial Architecture Review  
**Role**: Adversarial Breaker  
**Repository**: `/home/rishil-n/AI_PROJECT`

---

## 1. CRITICAL FINDINGS

### C1. False Accuracy & Deceptive Fallback on LLM Judge Failure
* **Component**: `backend/app/gate/score.py` & `backend/app/plugins/judges/gemini_judge.py`
* **Problem**: In Section 5 Error Handling, when `GeminiJudgePlugin` times out or raises an API error during an `AMBIGUOUS` evaluation, the architecture proposes returning `GateResult(verdict="RESOLVED_AMBIGUOUS", resolved_by="llm_judge", explanation="Judge unavailable")`.
* **Why It Matters**: This presents a failed judge call as a successfully resolved judgment. A downstream API consumer or automated system reading `resolved_by: "llm_judge"` will assume the response was audited by Gemini, when in reality no LLM review occurred. This violates the core scientific honesty principle of the spec.
* **Recommendation**: If Stage 3 fails, the system must either return an explicit HTTP 502/503 error or output `verdict: "AMBIGUOUS"`, `resolved_by: "gate"`, `explanation_source: "symbolic"`, and explicitly flag `judge_status: "errored"`. Never return `resolved_by: "llm_judge"` when the judge failed.

### C2. Unsupervised Cluster-Label Alignment Hazard (Stage 1 Fingerprinting)
* **Component**: `backend/app/fingerprint/cluster.py`
* **Problem**: Section M4 specifies running k-means ($k=2$) on feature vectors to identify `["hallucination_region", "correct_region"]`. Standard k-means is an unsupervised algorithm that assigns arbitrary cluster IDs (`0` and `1`). The architecture does not specify how cluster `0` is mapped to `"hallucination_region"` vs `"correct_region"`.
* **Why It Matters**: On different models or runs, cluster `0` could correspond to correct responses for Model A, but hallucinated responses for Model B. Without ground-truth label alignment, the cluster centroids will be inverted 50% of the time, corrupting the Fingerprint Radar visualization and downstream logic.
* **Recommendation**: Either (a) calculate class-conditional centroids directly from the ground-truth binary labels in the training set (supervised centroids), or (b) align unsupervised k-means clusters by computing the mean hallucination label density per cluster.

---

## 2. HIGH-PRIORITY FINDINGS

### H1. Unhandled Uncalibrated Model Null-Pointer Crash
* **Component**: `backend/app/gate/score.py` & `backend/app/schemas.py`
* **Problem**: The `Fingerprint` schema defines `t_low: float | None` and `t_high: float | None`. If `/api/v1/score` is invoked for a model that has undergone Stage 1 clustering but not Stage 2 threshold calibration, `t_low` and `t_high` will be `None`.
* **Why It Matters**: Passing `None` to `classify(G, t_low, t_high)` will execute `if G < t_low:`, throwing an unhandled `TypeError: '<' not supported between instances of 'float' and 'NoneType'` in Python, causing an unhandled HTTP 500 server crash.
* **Recommendation**: Enforce an explicit validation check before scoring: if `fingerprint.t_low is None` or `fingerprint.t_high is None`, immediately raise `UncalibratedModelError` (mapping to HTTP 400 Bad Request / 404 Not Found), refusing to score uncalibrated models.

### H2. Secret Key Exposure Vulnerability in Error Tracebacks
* **Component**: `backend/app/api/routes_score.py` & `GeminiJudgePlugin`
* **Problem**: When `google-generativeai` throws exceptions (e.g., `GoogleAPIError`, `HTTPError` due to invalid keys or quota limits), the exception text frequently embeds the active API key or authorization headers.
* **Why It Matters**: If API route exception handlers catch generic `Exception` and pass `str(e)` to HTTP response payloads or log streams returned to the client, the `GEMINI_API_KEY` will be leaked to external API consumers.
* **Recommendation**: Wrap all Gemini API calls in a dedicated try-except block that catches SDK exceptions, strips authorization headers/keys, logs generic error codes internally, and returns sanitized, user-safe error messages to the API layer.

### H3. Empty `ambiguous_patterns` Prompt Breakdown in Stage 3
* **Component**: `backend/app/gate/score.py` & `backend/app/plugins/judges/gemini_judge.py`
* **Problem**: The judge prompt relies on receiving `ambiguous_patterns: list[dict]`. However, a response score $G = \sum w_i r_i$ can land in the ambiguous range $[T_L, T_H]$ via the cumulative sum of multiple weak rule activations ($r_i < \text{threshold}$), resulting in `triggered_patterns` being empty (`[]`).
* **Why It Matters**: If `ambiguous_patterns` is empty, the LLM Judge prompt will format an empty list ("Triggered Rules: None"), causing the prompt template to fail or providing no actionable symbolic context to the LLM judge.
* **Recommendation**: Handle empty `triggered_patterns` gracefully in prompt formatting by providing a fallback summary of feature deviations (e.g., "No single rule pattern crossed full threshold, but cumulative score G=0.65 fell in ambiguous range").

---

## 3. MEDIUM FINDINGS

### M1. Confounded LLM Judge via Model Identity Leakage
* **Component**: `backend/app/plugins/judges/gemini_judge.py`
* **Problem**: The prompt specification states that `GeminiJudgePlugin` receives "the model's fingerprint summary". If this summary includes the literal `model_id` (e.g., `"chatgpt"`, `"llama-3"`), the LLM judge will utilize its parametric priors about that specific model's historical reliability rather than evaluating the linguistic telemetry of the answer text.
* **Why It Matters**: This introduces a massive research confounder. The judge's performance will reflect its internal pre-trained biases about models rather than the effectiveness of the neuro-symbolic gate.
* **Recommendation**: Strip `model_id` and model names from the payload sent to `JudgePlugin`. Present the model anonymously as `Model-X` with anonymized numerical centroid features.

### M2. Degenerate & Out-of-Bounds Threshold Handling
* **Component**: `backend/app/fingerprint/calibrate.py` & `backend/app/gate/score.py`
* **Problem**: Grid search during threshold fitting can produce edge cases where $T_L = T_H$, $T_L > T_H$, $T_L < 0$, or $T_H > 1$.
* **Why It Matters**: 
  - If $T_L = T_H$, the ambiguous band collapses to a 0-width point. Any exact match ($G == T_L$) triggers `AMBIGUOUS`, while all other points bypass the judge.
  - If thresholds are unconstrained, $T_H$ could be $> 1.0$, rendering `HIGH_RISK` impossible to trigger.
* **Recommendation**: Enforce strict boundary constraints during calibration: $0.0 \le T_L \le T_H - \epsilon \le 1.0$, where $\epsilon \ge 0.05$ guarantees a minimum non-zero ambiguous band width.

### M3. Hardcoded Single-Model Ingestion Assumption
* **Component**: `backend/app/plugins/datasets/halueval_loader.py`
* **Problem**: Section M1 assigns a fixed `model_id = "chatgpt"` to all records loaded from HaluEval's `qa_data.json`.
* **Why It Matters**: If HaluEval QA or custom evaluation datasets contain outputs generated by multiple LLMs, hardcoding `model_id` corrupts per-model fingerprint isolation and prevents testing per-model adaptivity across distinct generating models.
* **Recommendation**: Inspect the raw dataset schema dynamically for generator metadata; if absent, make `default_model_id` a configurable parameter in `DatasetLoaderPlugin`.

### M4. In-Memory Job Store Data Loss
* **Component**: `backend/app/jobs/background.py`
* **Problem**: Background calibration jobs store status and results in a plain Python in-memory dictionary (`_JOBS: dict[str, dict]`).
* **Why It Matters**: Any server restart, uvicorn worker reload, or crash will instantly wipe all active and completed job history. Clients polling `GET /api/v1/jobs/{job_id}` will receive HTTP 404 errors for completed background runs.
* **Recommendation**: Persist job metadata to lightweight JSON files on disk (e.g. `data/processed/jobs/{job_id}.json`) so state survives process restarts.

---

## 4. DESIGN IMPROVEMENTS

1. **Decouple Gate Evaluation from Escalation Execution**: `score.py` should expose a pure function `evaluate_gate(features, fingerprint) -> GateEvaluation` that contains zero side effects or judge calls. Escalation should be a separate pipeline step, making unit testing fast and offline baseline evaluation clean.
2. **Asynchronous CPU Offloading for Embeddings**: Computing Feature D (semantic drift) using `sentence-transformers` on CPU during synchronous FastAPI request handling will block the asyncio event loop under load. Wrap embedding inference in `asyncio.to_thread()` or run it via a process pool worker.
3. **Structured Direction in API Schema**: Add an optional `resolved_direction: Literal["LOW_RISK", "HIGH_RISK"] | None` field to `GateResult` so API clients do not need to parse free-text explanations to determine the judge's recommendation.

---

## 5. QUESTIONS FOR HUMAN DECISION

1. **Stage 3 Judge Escalation Fallback**: If the Gemini API call fails due to quota or network errors, should the system return a `502 Bad Gateway` error, or return a non-escalated gate verdict with an error warning flag?
2. **Model Identity Anonymization**: Should model names be strictly stripped from all judge prompts to prevent LLM prior bias during research evaluation?
3. **Stage 1 Cluster Label Alignment**: Should Stage 1 fingerprint centroids be computed using supervised ground-truth class means (guaranteeing correct label alignment) or via unsupervised k-means with a post-hoc label matching step?
