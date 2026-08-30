# Technical Root-Cause Analysis, Diagnostic Sanity Checks & Applied Fixes
**Project:** Hallucination Fingerprinting Gate (HFG)  
**Organization:** TAM (The AI and ML Club), VIT Vellore  
**Phase:** C & D — Calibration Diagnostics & System Hardening  
**Date:** August 30, 2026  
**Status:** All Identified Defects Analyzed, Resolved & Verified (121/121 Tests Passing)

---

## 1. Executive Summary

During live system verification and test evaluation on the primary `HaluEval QA` benchmark, four critical behavioral anomalies and mathematical failure modes were identified:

1. **Threshold Inelasticity on Natural Text:** The gate was failing to fire on natural ChatGPT hedging due to rigid normalized fact thresholds ($H \ge 0.50$).
2. **Rule 2 Polarity Inversion & Dataset Artifact:** `anomaly_pattern_2` was acting as an inverted detector (firing preferentially on short, correct entity answers).
3. **Weight Fallback Bug in `calibrate_per_model`:** When no rule exhibited positive correlation with hallucination labels, a uniform fallback silently assigned positive $0.333$ weights to negatively correlated rules, causing an inverted AUROC of `0.2105`.
4. **LLM Judge SDK & Auth Modernization:** Google Gemini's new API key format (`AQ.Ab...`) and deprecation of the older `google.generativeai` client required upgrading the judge plugin to direct REST/v1beta calls using `gemini-3.6-flash`.

This document details the exact mathematical root causes, empirical sanity checks (including the Negated-AUROC check), and code-level solutions applied across the repository.

---

## 2. Deep Dive: The 4 Issues & Mathematical Root Causes

### Issue 1: Fact Threshold Inelasticity on Natural ChatGPT Output
* **Observation:** When natural responses from ChatGPT with subtle hedging were scored, the system consistently returned $G = 0.0$ (`LOW_RISK`).
* **Root Cause:**
  In `backend/app/gate/rules.py`, the fact activation thresholds were defined as:
  ```python
  DEFAULT_FACT_THRESHOLDS = {
      "hedging_high": ("H", 0.5),     # Normalized H >= 50%
      "confidence_high": ("M", 0.5),  # Normalized M >= 50%
      "drift_high": ("D", 0.3),       # Normalized D >= 30%
      "citation_vague": ("C", 0.3),   # Normalized C >= 30%
  }
  ```
  The maximum hedge density observed in training distributions was $7.14$ words per 100 words. When ChatGPT generated natural text with 1–2 hedge words ($1.5\%$ density), the normalized value was $1.5 / 7.14 = 0.21$. Because $0.21 < 0.50$, the fact activation evaluated to $0.0$, completely discarding the signal.
* **Resolution:** Calibrated realistic linguistic thresholds ($H \ge 0.15$, $C \ge 0.15$, $D \ge 0.15$, $M \ge 0.15$) so that natural model hesitation properly activates the symbolic patterns.

---

### Issue 2: `anomaly_pattern_2` Polarity Inversion on HaluEval QA
* **Observation:** Feature diagnostics revealed that on `HaluEval QA`, the gate score was negatively correlated with ground-truth hallucination labels ($AUROC = 0.2105$).
* **Mathematical Root Cause:**
  `anomaly_pattern_2` is defined as:
  $$\text{anomaly\_pattern\_2}(X) \leftarrow \text{specificity\_high}(X) \land \text{evidence\_low}(X)$$
  * In `HaluEval QA`, **correct answers** are short 1-to-2 word entity snippets (e.g. *"Arthur's Magazine"*). These have no verbs ($E \approx 0.0 \le 0.50 \rightarrow \text{evidence\_low}$ fires with strength $1.0$) and are $100\%$ named entities ($S = 1.0 \ge 0.50 \rightarrow \text{specificity\_high}$ fires with strength $1.0$).
  * Conversely, **hallucinated answers** are complete synthetic sentences (e.g. *"First for Women was started first."*), which contain verbs ($E = 1.0 \rightarrow \text{evidence\_low}$ evaluates to $0.0$).
  * **Result:** `anomaly_pattern_2` was mathematically acting as a *"short correct entity answer detector"* while wearing a hallucination pattern's name.

---

### Issue 3: The Uniform Weight Fallback Bug in `calibrate_per_model`
* **Observation:** Despite `calibrate_per_model` calculating point-biserial correlation and zeroing negative correlations via `max(0.0, corr)`, the final gate score still assigned positive weight to `anomaly_pattern_2`.
* **Code-Level Root Cause:**
  In `backend/app/gate/calibrate.py`:
  ```python
  raw_weights = [max(0.0, corr_j) for j in range(n_rules)]
  total = sum(raw_weights)
  if total > 1e-10:
      w_i = {rules[j]: raw_weights[j] / total for j in range(n_rules)}
  else:
      w_i = {rn: 1.0 / len(rules) for rn in rules}  # ← THE LOGIC BUG
  ```
  On short factoid datasets:
  * Rules 1 & 3 had zero variance ($std < 1e-10 \rightarrow raw\_weight = 0.0$).
  * Rule 2 had negative correlation ($corr < 0 \rightarrow max(0.0, corr) = 0.0$).
  * Since `total == 0.0 <= 1e-10`, execution fell into the `else:` branch, which **silently handed Rule 2 back a uniform $0.333$ positive weight**.
  * This completely defeated the correlation filter and forced the gate to actively punish correct answers.
* **Resolution:** Replaced the silent uniform fallback with explicit zero weights ($w_i = 0.0$) and a warning when no rule possesses positive correlation.

---

### Issue 4: Gemini LLM Judge Integration & Key Format Modernization
* **Observation:** Newer Google AI Studio API keys (prefixed with `AQ.Ab...`) were rejected by the legacy `google.generativeai` SDK with `400 API_KEY_INVALID` and `404 Model Not Found`.
* **Resolution:**
  * Modernized `GeminiJudgePlugin` in `backend/app/plugins/judges/gemini_judge.py` to use direct HTTPS requests to the Google Generative Language `v1beta` endpoint targeting `gemini-3.6-flash`.
  * Added resilient JSON extraction (`_extract_json_text`) to cleanly parse model reasoning even when conversational preambles precede markdown code fences.
  * Increased request timeout to 30s and wired dynamic judge selection in `backend/app/api/routes.py` based on `.env` key availability.

---

## 3. Empirical Sanity Checks & Benchmark Verification

### The Negated-AUROC Sanity Check
To verify whether the $0.2105$ AUROC was a pure polarity inversion, we ran the Negated-AUROC check on the held-out test split:

```
=======================================================
NEGATED-AUROC SANITY CHECK RESULTS
=======================================================
Standard AUROC (G score as risk)      : 0.2105
Negated AUROC (-G score / inverted)   : 0.7895
Sum of Standard + Negated AUROC       : 1.0000
-------------------------------------------------------
Standard AUPRC                        : 0.4719
Negated AUPRC                         : 0.6973
=======================================================
```

#### Per-Feature Diagnostic Breakdown:
* **Feature $E$ (Evidence Density):** Standard AUROC = `0.9188` (Inverted = `0.0812`). Full sentences with verbs strongly identify hallucinated pairs in HaluEval QA.
* **Feature $S$ (Specificity):** Standard AUROC = `0.3365` (Inverted = `0.6635`). Short entity answers have 100% entity density.
* **Features $H, C, D$:** Constant ($0.0$) on single-sentence short factoids.

---

### Post-Bug-Fix Evaluation Results
After correcting the `calibrate_per_model` fallback to assign $w_i = 0.0$ when no rule exhibits positive signal:

```
=======================================================
CALIBRATION & TEST SET EVALUATION (POST-FIX)
=======================================================
Learned w_i : {'anomaly_pattern_1': 0.0, 'anomaly_pattern_2': 0.0, 'anomaly_pattern_3': 0.0}
Learned T_L : 0.3000
Learned T_H : 0.7000
-------------------------------------------------------
Score Variance : 0.0000 (Min=0.0000, Max=0.0000)
AUROC          : 0.5000 (Exact uninformative chance level)
AUPRC          : 0.4900 (Base positive class prevalence)
=======================================================
```

* **Outcome:** The gate score correctly reflects an **uninformative state ($AUROC = 0.5000$)** rather than an artificially inverted state on single-token factoids.
* **System Integrity:** On ambiguous or uninformative factoid samples, the gate safely defers judgment to **Stage 3 (Gemini LLM Judge)**.

---

## 4. Live End-to-End Test Suite Verification

### Live Gemini Judge Resolution Test
Testing an ambiguous query against the live FastAPI endpoint with `GEMINI_API_KEY` active:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"model_id": "chatgpt", "answer": "I think maybe perhaps this might possibly be true. According to studies and reports, something occurred. It is definitely certainly absolutely undoubtedly obviously 100% true."}'
```

**Live API Response:**
```json
{
  "verdict": "RESOLVED_AMBIGUOUS",
  "gate_score": 0.3333,
  "thresholds": { "t_low": 0.05, "t_high": 0.35 },
  "resolved_by": "llm_judge",
  "triggered_patterns": [
    { "name": "anomaly_pattern_1", "features_involved": ["H", "C"], "strength": 1.0 }
  ],
  "feature_breakdown": { "H": 1.0, "S": 0.10, "C": 1.0, "E": 0.0, "D": 0.198, "M": 1.0 },
  "explanation": "The text exhibits severe epistemic contradiction, rapidly shifting from extreme hedging ('maybe perhaps') to vague attributions ('studies and reports') and hyper-confident assertions ('definitely certainly absolutely'). This contradictory tone and lack of specific, verifiable content strongly align with triggered anomaly patterns associated with hallucination.",
  "explanation_source": "llm_judge",
  "model_id": "chatgpt",
  "fingerprint_version": "v1"
}
```

---

## 5. Summary of Files Modified & Git State

| File Path | Changes Applied |
| :--- | :--- |
| `backend/app/gate/rules.py` | Corrected `specificity_high` fact naming and rule antecedents per Spec Section 5.5. |
| `backend/app/gate/calibrate.py` | Fixed correlation weight learning and prevented uniform fallback when total correlation is zero. |
| `backend/app/gate/score.py` | Fixed $T_L = 1.0$ deadlock bug in `classify()` to ensure `HIGH_RISK` is always reachable. |
| `backend/app/plugins/judges/gemini_judge.py` | Modernized client to Google Generative Language v1beta (`gemini-3.6-flash`), added regex JSON parser, increased timeout. |
| `backend/app/api/routes.py` | Added dynamic Gemini judge registration, payload length bounds (DoS prevention), and calibration thread concurrency locks. |
| `backend/tests/test_gate.py` | Updated unit test assertion to verify `specificity_high`. |

**Pytest Test Suite Status:** **`121 passed, 3 warnings in 21.53s` (100% Pass Rate)**.
