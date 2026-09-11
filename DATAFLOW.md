# System Dataflow Architecture (DATAFLOW)

## Project: Hallucination Fingerprinting Gate (HFG)
### Subtitle: Complete Dataflow, State Transitions, and Sequence Architecture
**Document Version:** 2.0  
**Status:** Implemented & Verified  
**Target Audience:** System Architects, Backend Engineers, Frontend Engineers, ML Researchers  

---

## 1. High-Level System Dataflow Overview

The Hallucination Fingerprinting Gate processes information through three distinct execution flows:
1. **Real-Time Scoring Pipeline (Inference Dataflow)**: Synchronous evaluation of LLM answers through the 3-stage gate.
2. **Model Calibration Pipeline (Training / Profiling Dataflow)**: Asynchronous computation of behavioral fingerprints and optimal decision thresholds.
3. **Evaluation Benchmark Pipeline (Experimentation Dataflow)**: Offline evaluation of baseline models and computation of AUROC / AUPRC / ECE metrics.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 HIGH-LEVEL DATA PIPELINE                               │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [ Raw Answer Text ] + [ Model ID ]
              │
              ▼
    ┌─────────────────────────┐
    │ Stage 1: Telemetry      │ ──► Extracts 6 raw features: [H, S, C, E, D, M]
    │ Feature Extraction      │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Feature Normalization   │ ──► Scaled to [0, 1] using Model Fingerprint Min/Max
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Stage 2: Neuro-Symbolic │ ──► Evaluates 3 relational rules via PyReason
    │ Decision Gate           │ ──► Computes aggregate score: G = Σ w_i * r_i
    └───────────┬─────────────┘
                │
        ┌───────┴──────────────────────────────┐
        │                                      │
   [ G < T_L or G > T_H ]               [ T_L ≤ G ≤ T_H ]
        │                                      │
        ▼ (Fast Path)                          ▼ (Escalation Path)
  LOW_RISK / HIGH_RISK                 ┌─────────────────────────┐
        │                              │ Stage 3: LLM Judge      │ (Gemini 1.5/2.0 API)
        │                              │ Escalation              │ Injects ambiguous rules
        │                              └───────────┬─────────────┘
        │                                          │
        │                                          ▼
        │                                 RESOLVED_AMBIGUOUS
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              [ Unified ScoreResponse JSON ]
                           │
                           ▼
          [ Frontend "Signal Forensics" UI ]
```

---

## 2. Detailed Dataflow Pipelines

---

### Pipeline 1: Real-Time Answer Scoring Pipeline (`POST /api/v1/score`)

This flow processes an incoming candidate answer from an HTTP client and returns an explainable risk verdict.

```mermaid
flowchart TD
    Client([HTTP Client / Frontend]) -->|1. POST /api/v1/score\n{answer, model_id}| FastAPI[FastAPI Route Handler]
    
    FastAPI -->|2. Lookup Fingerprint| FPStore[(In-Memory Fingerprint Store)]
    FPStore -->|Model Fingerprint Found| FastAPI
    FPStore -.->|Model Not Found| Err[Raise UncalibratedModelError 404]
    
    FastAPI -->|3. Extract Telemetry| Extractor[DefaultFeatureExtractor]
    Extractor -->|Lexicon Match| F_H[H: Hedge Density]
    Extractor -->|SpaCy NER & Regex| F_S[S: Specificity]
    Extractor -->|Pattern Proximity| F_C[C: Citation Vagueness]
    Extractor -->|Entity Co-occurrence| F_E[E: Evidence Density]
    Extractor -->|SentenceTransformers| F_D[D: Semantic Drift]
    Extractor -->|Lexicon Match| F_M[M: Confidence Marker]
    
    F_H & F_S & F_C & F_E & F_D & F_M -->|Raw Feature Vector| Normalizer[Min-Max Normalizer]
    Normalizer -->|Normalized Vector [0, 1]^6| Gate[PyReason Neuro-Symbolic Gate]
    
    Gate -->|Evaluate Pattern 1| P1[Anomaly Pattern 1: Evasive]
    Gate -->|Evaluate Pattern 2| P2[Anomaly Pattern 2: Over-Specific]
    Gate -->|Evaluate Pattern 3| P3[Anomaly Pattern 3: Drift/Overconfidence]
    
    P1 & P2 & P3 -->|Weighted Sum G = Σ w_i * r_i| ScoreCalc[Gate Score G]
    
    ScoreCalc --> Branch{Evaluate G vs\n(T_L, T_H)}
    
    Branch -->|G < T_L| LowRisk[Verdict: LOW_RISK\nresolved_by: 'gate']
    Branch -->|G > T_H| HighRisk[Verdict: HIGH_RISK\nresolved_by: 'gate']
    
    Branch -->|T_L <= G <= T_H| Escalate[Escalate to Stage 3]
    
    Escalate -->|Synthesize Prompt with Rule Activations| Judge[GeminiJudgePlugin / MockJudge]
    Judge -->|Call Gemini API / Local Fallback| JudgeRes[LLM Judge Analysis]
    JudgeRes --> ResAmb[Verdict: RESOLVED_AMBIGUOUS\nresolved_by: 'llm_judge']
    
    LowRisk & HighRisk & ResAmb --> Aggregator[Response Synthesizer]
    Aggregator -->|JSON Response| Client
```

#### Step-by-Step Data Transformations (Inference)

| Stage | Data Input | Component | Output Data Representation |
| :--- | :--- | :--- | :--- |
| **Input** | Raw JSON | FastAPI Router | `ScoreRequest(answer=str, model_id=str)` |
| **Extraction** | Answer string | `DefaultFeatureExtractor` | `dict[str, float]`: `{H: 1.25, S: 0.42, C: 0.0, E: 0.67, D: 0.03, M: 0.85}` |
| **Normalization** | Raw dict + Model bounds | `normalize_features()` | `dict[str, float]` with values strictly bounded in `[0.0, 1.0]` |
| **Symbolic Gate** | Normalized dict | `score_answer()` (PyReason) | `G: float`, `triggered_patterns: list[dict]`, initial verdict |
| **Escalation** (if needed) | Text + Triggered Patterns | `GeminiJudgePlugin` | `JudgeResult(verdict=str, confidence=float, explanation=str)` |
| **Output** | Verdict + Telemetry | Pydantic Serializer | `ScoreResponse` JSON schema |

---

### Pipeline 2: Offline Model Calibration Pipeline (`POST /api/v1/models/{id}/calibrate`)

This asynchronous pipeline calculates the behavioral baseline for a model from training records.

```mermaid
flowchart TD
    Req([Admin / POST /api/v1/models/{id}/calibrate]) --> FastApi[FastAPI Router]
    FastApi --> JobInit[Initialize Job Entry: status='pending']
    FastApi --> Spawn[Spawn Background Worker Thread]
    FastApi -->|Return job_id immediately| Req
    
    Spawn --> LockCheck{Acquire _calibration_lock}
    LockCheck -->|Lock Acquired| JobRunning[Set Job Status: 'running']
    LockCheck -.->|Already Active| JobConflict[Mark status='failed': Conflict]
    
    JobRunning --> DataLoader[HaluEval QA Loader]
    DataLoader --> Filter[ensure_primary_only Filter]
    Filter -->|Filtered Records| BatchExtractor[Batch Feature Extractor]
    
    BatchExtractor --> Mat[Feature Matrix N x 6]
    
    Mat --> Cluster[K-Means / GMM Clustering]
    Cluster --> Centroids[Compute Factual & Hallucinated Centroids]
    
    Mat --> Splitter[Train / Val Splitter]
    Splitter --> GridSearch[Grid Search Threshold Optimizer]
    
    GridSearch -->|Optimize Objective L = AUROC - λ * EscRate| OptParams[Optimal T_L, T_H, w_i]
    
    Centroids & OptParams --> Assembler[Assemble Fingerprint Object]
    Assembler --> SaveMem[(In-Memory _fingerprints)]
    Assembler --> SaveDisk[(Persist to configs/fingerprints/*.json)]
    
    SaveMem & SaveDisk --> JobDone[Set Job Status: 'complete', result=summary]
    JobDone --> ReleaseLock[Release _calibration_lock]
```

---

### Pipeline 3: Evaluation Benchmark Pipeline (`GET /api/v1/eval/summary`)

This flow powers the research validation dashboard, comparing the adaptive gate against standard baseline algorithms.

```mermaid
flowchart TD
    EvalTrigger([CLI / API /eval/summary]) --> EvalHarness[eval/run_experiment.py]
    
    EvalHarness --> TestData[(Held-out Test Dataset)]
    
    TestData --> Model1[Evaluator: Adaptive Per-Model Gate]
    TestData --> Base1[Baseline 1: Always-Escalate Judge]
    TestData --> Base2[Baseline 2: Never-Escalate Gate-Only]
    TestData --> Base3[Baseline 3: Global Pooled Threshold]
    TestData --> Base4[Baseline 4: Random Classifier]
    TestData --> Base5[Baseline 5: Generic TRACT Lexical]
    
    Model1 & Base1 & Base2 & Base3 & Base4 & Base5 --> MetricsCalc[eval/metrics.py]
    
    MetricsCalc --> M_ROC[Compute AUROC]
    MetricsCalc --> M_PRC[Compute AUPRC]
    MetricsCalc --> M_ECE[Compute Expected Calibration Error]
    MetricsCalc --> M_ESC[Compute Escalation Rate]
    
    M_ROC & M_PRC & M_ECE & M_ESC --> Aggregator[Experiment Results Compiler]
    Aggregator --> JSONSink[(eval/results/*.json)]
    JSONSink --> APIEndpoint[FastAPI GET /api/v1/eval/summary]
    APIEndpoint --> FrontendEval[Frontend Evaluation Dashboard View]
```

---

## 3. Sequence Diagrams

### 3.1 Fast-Path Scoring Sequence (Clean or Obvious Hallucination)

When the anomaly score falls outside the ambiguous band ($G < T_L$ or $G > T_H$), the response resolves in a single rapid pass without external LLM calls.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Browser
    participant API as FastAPI Backend
    participant Ext as DefaultFeatureExtractor
    participant Gate as Neuro-Symbolic Gate (PyReason)
    
    User->>API: POST /api/v1/score {answer, model_id}
    Note over API: Lookup model fingerprint
    API->>Ext: extract(answer)
    Ext-->>API: raw_features {H, S, C, E, D, M}
    Note over API: Normalize features with model ranges
    API->>Gate: score(normalized_features, thresholds, weights)
    Note over Gate: Evaluate Anomaly Patterns (1, 2, 3)
    Note over Gate: G = Σ w_i * r_i (e.g., G = 0.18 < T_L = 0.35)
    Gate-->>API: GateResult(verdict=LOW_RISK, G=0.18, patterns=[])
    Note over API: resolved_by = 'gate' (No Judge Call)
    API-->>User: 200 OK ScoreResponse JSON
```

---

### 3.2 Escalated Scoring Sequence (Ambiguous Verdict)

When the anomaly score falls into the ambiguous band ($T_L \le G \le T_H$), the gate delegates to Stage 3 for deeper semantic analysis.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Browser
    participant API as FastAPI Backend
    participant Ext as DefaultFeatureExtractor
    participant Gate as Neuro-Symbolic Gate
    participant Judge as GeminiJudgePlugin
    participant Gemini as Google Gemini API
    
    User->>API: POST /api/v1/score {answer, model_id}
    API->>Ext: extract(answer)
    Ext-->>API: raw_features
    API->>Gate: score(normalized_features)
    Gate-->>API: GateResult(verdict=AMBIGUOUS, G=0.52, triggered_patterns=[Pattern 1])
    
    Note over API: G in [T_L, T_H] -> Initiate Escalation
    API->>Judge: judge(answer, fingerprint, ambiguous_patterns)
    Note over Judge: Construct prompt with targeted rule context
    Judge->>Gemini: POST generateContent (model: gemini-1.5-flash)
    Gemini-->>Judge: {verdict: "HIGH_RISK", explanation: "Lacks grounded evidence..."}
    Judge-->>API: JudgeResult(verdict=RESOLVED_AMBIGUOUS, explanation=...)
    
    Note over API: Synthesize full explainability packet
    API-->>User: 200 OK ScoreResponse (resolved_by='llm_judge')
```

---

### 3.3 Asynchronous Model Calibration Job Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Admin / Client
    participant API as FastAPI Router
    participant Worker as Background Thread
    participant Ingest as Data Ingestion
    participant Calib as Calibration Engine
    participant Store as In-Memory Store
    
    Admin->>API: POST /api/v1/models/{id}/calibrate
    API->>API: Generate job_id, set status='pending'
    API->>Worker: spawn_thread(run_calibration, job_id, model_id)
    API-->>Admin: 202 Accepted {job_id: "job-123", status: "pending"}
    
    Worker->>Worker: Acquire _calibration_lock
    Worker->>Ingest: load(HaluEval-QA)
    Ingest-->>Worker: Primary Records
    Worker->>Calib: cluster_and_calibrate(records)
    Note over Calib: Compute Centroids & Grid Search (T_L, T_H, w_i)
    Calib-->>Worker: Fingerprint(model_id, thresholds, weights)
    Worker->>Store: _fingerprints[model_id] = fingerprint
    Worker->>Worker: Update job: status='complete'
    Worker->>Worker: Release _calibration_lock
    
    loop Poll Job Status
        Admin->>API: GET /api/v1/jobs/job-123
        API-->>Admin: {status: "complete", result: {model_id: "...", t_low: 0.35, t_high: 0.68}}
    end
```

---

## 4. State Transition Models

### 4.1 Gate Score Decision State Machine

```mermaid
stateDiagram-v2
    [*] --> IngestText: Incoming Answer Text
    IngestText --> FeatureExtraction: Extract 6 Features
    FeatureExtraction --> Normalization: Scale with Model Bounds
    Normalization --> SymbolicGate: Evaluate PyReason Rules
    
    SymbolicGate --> LowRisk: G < T_L
    SymbolicGate --> HighRisk: G > T_H
    SymbolicGate --> Ambiguous: T_L <= G <= T_H
    
    LowRisk --> TerminalVerdict: Direct Gate Resolution
    HighRisk --> TerminalVerdict: Direct Gate Resolution
    
    Ambiguous --> JudgeEscalation: Trigger Gemini Judge
    JudgeEscalation --> ResolvedAmbiguous: Judge Resolves Risk
    ResolvedAmbiguous --> TerminalVerdict: Annotated Resolution
    
    TerminalVerdict --> [*]
```

### 4.2 Background Calibration Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Pending: POST /calibrate
    Pending --> Running: Thread Spawned & Lock Acquired
    Pending --> Failed: Mutex Collision / Lock Timeout
    Running --> Complete: Calibration & Optimization Succeeded
    Running --> Failed: Dataset Empty / Insufficient Variance
    Complete --> [*]
    Failed --> [*]
```

---

## 5. Detailed Schema & Payload Specifications

### 5.1 Request Payload: `POST /api/v1/score`
```json
{
  "answer": "The Apollo 11 mission was definitely the most celebrated lunar landing, reportedly confirmed by studies show in 1969.",
  "model_id": "meta-llama/Llama-2-7b-chat-hf"
}
```

### 5.2 Intermediate Telemetry Feature Extraction Vector
```json
{
  "H": 0.0,
  "S": 0.1875,
  "C": 1.0,
  "E": 0.5,
  "D": 0.0,
  "M": 5.55
}
```

### 5.3 Normalized Vector ($[0, 1]^6$)
```json
{
  "H": 0.0,
  "S": 0.28,
  "C": 0.50,
  "E": 0.55,
  "D": 0.0,
  "M": 0.86
}
```

### 5.4 Final Response Payload: `ScoreResponse`
```json
{
  "verdict": "RESOLVED_AMBIGUOUS",
  "gate_score": 0.54,
  "thresholds": {
    "t_low": 0.32,
    "t_high": 0.65
  },
  "resolved_by": "llm_judge",
  "triggered_patterns": [
    {
      "name": "anomaly_pattern_1",
      "features_involved": ["citation_vague", "hedging_high"],
      "strength": 0.50
    }
  ],
  "feature_breakdown": {
    "H": 0.0,
    "S": 0.1875,
    "C": 1.0,
    "E": 0.5,
    "D": 0.0,
    "M": 5.55
  },
  "explanation": "The response makes an unanchored gesture to authority ('studies show') without citing a credible historical or scientific publication.",
  "explanation_source": "llm_judge",
  "model_id": "meta-llama/Llama-2-7b-chat-hf",
  "fingerprint_version": "v1.0.0"
}
```
