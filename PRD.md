# Product Requirements Document (PRD)

## Project: Hallucination Fingerprinting Gate (HFG)
### Subtitle: Model-Calibrated Neuro-Symbolic Decision Gate for Adaptive Hallucination Detection
**Document Version:** 2.0  
**Status:** Approved & Implemented  
**Target Audience:** AI Safety Researchers, ML Engineers, Technical Evaluators, Hackathon / Demo Reviewers  

---

## 1. Executive Summary

Large Language Models (LLMs) hallucinate differently. A model trained with aggressive reinforcement learning from human feedback (RLHF) might produce confident, citation-heavy fabrications, while another model might exhibit excessive hedging and syntactic drift when recalling uncertain knowledge. 

Existing hallucination detection solutions suffer from two major flaws:
1. **One-Size-Fits-All Thresholds**: They apply universal classification boundaries regardless of the architecture or behavioral profile of the generating model.
2. **Extreme Cost or Opacity**: They either require expensive, full-context LLM-as-a-judge calls on every query, or rely on white-box internal activations (logits, hidden embeddings) that are inaccessible behind commercial proprietary APIs.

The **Hallucination Fingerprinting Gate (HFG)** is a research-grade neuro-symbolic framework that treats hallucination detection as a **per-model behavioral phenomenon**. By extracting black-box linguistic telemetry (hedging, specificity, citation vagueness, evidence density, semantic drift, and confidence markers), evaluating them against deterministic symbolic rules with **per-model calibrated thresholds**, and escalating **only ambiguous edge-cases** to an LLM judge (Google Gemini), HFG optimizes the accuracy-vs-cost Pareto frontier while delivering interpretable, forensic explanations.

---

## 2. Problem Statement & Research Gap

### 2.1 The Problem
When evaluating factual accuracy in real-time AI applications:
- Invoking an LLM judge for 100% of queries introduces prohibitive latency (1–4 seconds) and operational token costs.
- Traditional heuristics (lexicon matching, perplexity) generate high false-positive rates because they fail to adapt to individual model baseline styles.
- Internal-state methods (e.g., activation probes, entropy checks) require white-box model weights and token log-probabilities, which are unavailable for black-box models (e.g., GPT-4o, Claude 3.5, Gemini 1.5).

### 2.2 The Research Gap
| Approach | Access Needed | Interpretable? | Per-Model Adaptive? | Cost Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Direct LLM Judge** | Black-box | High (CoT) | No (Static prompt) | Very Poor (100% LLM calls) |
| **White-Box Probes (LLM-Check)** | White-box (activations) | No (Opaque vectors) | Yes | High |
| **Self-Consistency (SelfCheckGPT)** | Black-box (N samples) | Medium | No | Very Poor (N stochastic calls) |
| **Global Heuristic Gate** | Black-box | Yes | No (Pooled threshold) | High |
| **Hallucination Fingerprinting Gate (Ours)** | **Black-box (Text only)** | **Yes (Symbolic rules)** | **Yes (Learned $T_L, T_H, w_i$)** | **Optimal (Escalate only ambiguous)** |

---

## 3. Product Vision & Value Proposition

### 3.1 Vision
To provide a fast, interpretable, and mathematically calibrated gatekeeper that sits between LLM outputs and end-users, filtering out hallucinations at near-zero incremental token cost while providing forensic visual explanations.

### 3.2 Core Value Proposition
- **Black-Box Operation**: Operates purely on raw input prompt and generated answer text.
- **Per-Model Calibration**: Learns distinct decision boundaries $(T_L, T_H)$ and rule weights $w_i$ for each supported LLM.
- **Cost Reduction**: Rejects obvious truths ($G < T_L$) and flags obvious fabrications ($G > T_H$) instantly, cutting expensive LLM judge calls by 50% to 80%.
- **Explainability as Evidence**: Explanations cite exact linguistic patterns and feature violations rather than unverifiable LLM chain-of-thought fabrications.

---

## 4. User Personas & Use Cases

### 4.1 Target Personas
1. **AI Safety & Alignment Researcher**: Needs to benchmark different foundation models on factual consistency, measure stylistic tells, and inspect Pareto frontiers of detection accuracy vs. escalation rate.
2. **LLM Systems Architect / Production Engineer**: Seeks a low-latency screening middleware to reject hallucinations before routing outputs to end consumers or downstream execution agents.
3. **Enterprise AI Auditor**: Requires auditable, rule-based provenance explaining *why* an LLM response was flagged or passed, without relying on black-box judge hallucinations.

### 4.2 Primary Use Cases
- **Real-Time Answer Scoring**: An API consumer submits an answer string and model identifier to receive an instant risk verdict (`LOW_RISK`, `HIGH_RISK`, or `RESOLVED_AMBIGUOUS`) with feature telemetry.
- **Model Fingerprint Exploration**: A user browses calibrated models in the UI to inspect radar profiles displaying behavioral baseline distributions.
- **Comparative Evaluation & Benchmarking**: Evaluating adaptive vs. pooled thresholds across benchmark datasets (HaluEval QA, TruthfulQA, SimpleQA) to prove performance gains.

---

## 5. Product Features & Functional Requirements

### 5.1 Stage 1: Behavioral Telemetry Extraction (FR-1)
The system must deterministically extract 6 linguistic features from raw answer text without making any external API or LLM calls:
- **FR-1.1 (H - Hedge Density)**: Measure frequency of linguistic qualifiers ("might", "perhaps", "could be") per 100 words.
- **FR-1.2 (S - Specificity)**: Measure proportion of named entities, dates, and numbers relative to total token count using SpaCy NER.
- **FR-1.3 (C - Citation Vagueness)**: Detect gestures toward authority ("studies show", "according to experts") lacking proximal named attribution.
- **FR-1.4 (E - Evidence Density)**: Calculate concrete relational assertions per sentence via entity-relation co-occurrence heuristics.
- **FR-1.5 (D - Semantic Drift)**: Calculate pairwise cosine embedding variance across sequential sentences using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **FR-1.6 (M - Confidence Marker Density)**: Measure frequency of emphatic assertions ("definitely", "guaranteed", "unquestionably") per 100 words.
- **FR-1.7 (Per-Model Normalization)**: Normalize feature vectors into the range $[0, 1]$ using per-model min/max scaling parameters derived from offline calibration.

### 5.2 Stage 2: Neuro-Symbolic Decision Gate (FR-2)
- **FR-2.1 (Symbolic Rule Graph)**: Implement deterministic rules (via PyReason framework) mapping normalized feature intervals to anomaly patterns:
  - *Pattern 1*: High hedging + High citation vagueness.
  - *Pattern 2*: High specificity + Low evidence density.
  - *Pattern 3*: High confidence markers + High semantic drift.
- **FR-2.2 (Gate Score Formulation)**: Compute the composite anomaly gate score:
  $$G = \sum_{i=1}^{3} w_i \cdot r_i$$
  where $r_i \in [0, 1]$ is the rule activation strength and $w_i$ is the calibrated rule weight ($\sum w_i = 1$).
- **FR-2.3 (Three-Tier Verdict Classification)**:
  - $G < T_L \implies$ **LOW RISK** (Clean output accepted immediately; no escalation).
  - $G > T_H \implies$ **HIGH RISK** (Flagged as hallucination; no escalation).
  - $T_L \le G \le T_H \implies$ **AMBIGUOUS** (Requires Stage 3 LLM Judge escalation).

### 5.3 Stage 3: LLM Judge Escalation (FR-3)
- **FR-3.1 (Targeted Context Injection)**: When a response is classified as `AMBIGUOUS`, synthesize a structured prompt containing the question, answer, model profile, and the specific rule pattern that triggered the ambiguity.
- **FR-3.2 (Gemini API Integration)**: Call the Gemini API to resolve the ambiguity and output a final verdict (`RESOLVED_AMBIGUOUS`) with natural-language reasoning.
- **FR-3.3 (Mock Judge Fallback)**: Provide a deterministic `MockJudgePlugin` that functions seamlessly without external network access or API keys.

### 5.4 Offline Calibration & Fingerprint Construction (FR-4)
- **FR-4.1 (K-Means Clustering)**: Cluster feature distributions into hallucinated vs. factual centroids per model.
- **FR-4.2 (Grid-Search Threshold Calibration)**: Automatically determine optimal model-specific boundaries $(T_L, T_H)$ and weights $(w_1, w_2, w_3)$ on a validation split.
- **FR-4.3 (Dataset Role Enforcement)**: Ensure calibration strictly consumes **primary** data (HaluEval QA), raising clear warnings or errors if secondary datasets (TruthfulQA, SimpleQA) are introduced into calibration splits.

### 5.5 "Signal Forensics" User Interface (FR-5)
- **FR-5.1 (Scan Report View)**: Deliver a clinical, lab-report styled interface displaying raw inputs, telemetry bar charts, gate threshold lines, and final verdicts.
- **FR-5.2 (Fingerprint Radar Component)**: Render a 6-axis polar radar chart modeling the model's calibrated normal region as soft concentric ridges, overlaying the active response's feature trajectory.
- **FR-5.3 (Model Explorer)**: Provide a catalog view of all registered models with their calibration timestamps, sample counts, and threshold bands.
- **FR-5.4 (Evaluation Dashboard)**: Display experimental benchmark tables (AUROC, AUPRC, ECE, escalation rates) comparing adaptive thresholds against global baselines.
- **FR-5.5 (Persistent Research Disclaimer)**: Present an un-dismissible notification stating that the system is a research demonstration and not an infallible ground-truth fact-checker.

---

## 6. Non-Functional Requirements (NFRs)

### 6.1 Performance & Latency
- **NFR-1.1**: Gate-only scoring ($G < T_L$ or $G > T_H$) must resolve in **< 1.5 seconds** on CPU.
- **NFR-1.2**: Escalate-to-judge path must complete within **< 4.0 seconds** subject to upstream LLM API latency.
- **NFR-1.3**: Single-port architecture: FastAPI must serve both API endpoints and static pre-built React frontend assets from a single port (`8000`).

### 6.2 Modularity & Extensibility
- **NFR-2.1**: Extensible plugin architecture for Feature Extractors, Judges, and Dataset Loaders via strict Python protocols.
- **NFR-2.2**: Swapping components must occur purely via configuration without code modifications at call sites.

### 6.3 Reliability & Safety
- **NFR-3.1**: Graceful degradation: If Gemini API fails or lacks credentials, the system must either fallback to Mock Judge or raise clear domain exceptions (`GeminiJudgeError`, `UncalibratedModelError`).
- **NFR-3.2**: Thread-safe calibration: Ensure concurrent calibration requests on the same model ID are safely locked and rejected.

---

## 7. Scope Boundaries & Exclusions

### 7.1 In-Scope (v1 / v2 Prototype)
- Response-level factual hallucination detection for single-turn factual QA.
- Six core telemetry features (H, S, C, E, D, M).
- Per-model threshold calibration on HaluEval QA.
- REST API (7 endpoints) and React frontend with "Signal Forensics" visual system.
- One-click launcher scripts for developer and Windows distribution.

### 7.2 Out-of-Scope (Explicitly Deferred)
- **Claim-Level Token Decomposition**: Breaking long-form responses into individual atomic claims and highlighting specific sentences (deferred to future versions).
- **Multi-Tenant User Management & Billing**: No authentication, API rate-limiting tiers, or user database.
- **External Knowledge Retrieval**: No active RAG or web-search fact verification against Wikipedia/Wikidata (the system evaluates linguistic behavior, not external truth tables).
- **Adversarial Robustness**: The system is not certified against prompt injection or adversarial text manipulation designed to fool lexicon density checks.

---

## 8. Success Metrics & Key Performance Indicators (KPIs)

| Metric | Target | Verification Method |
| :--- | :--- | :--- |
| **AUROC Gain** | $\ge +0.04$ over global pooled baseline | `eval/run_experiment.py` test harness |
| **Escalation Rate** | $\le 25\%$ of total queries | Measured on held-out HaluEval test split |
| **Test Suite Quality** | 100% pass rate across 121 unit tests | Pytest execution in automated CI |
| **Gate Latency** | $< 1.5$ seconds (P95) | Synchronous benchmark on CPU |
| **Explainability Coverage** | 100% of verdicts cite triggered rules or telemetry | API response contract validation |
