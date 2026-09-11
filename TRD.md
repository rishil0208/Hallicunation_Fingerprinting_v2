# Technical Requirements Document (TRD)

## Project: Hallucination Fingerprinting Gate (HFG)
### Subtitle: Model-Calibrated Neuro-Symbolic Decision Gate Architecture
**Document Version:** 2.0  
**Status:** Implemented & Verified  
**Target Audience:** Core Developers, System Architects, DevOps / Deployment Engineers  

---

## 1. System Architecture & Tech Stack

### 1.1 Technology Stack
- **Runtime & Language**: Python 3.10+ (tested on Python 3.10.21 Linux x86_64).
- **Backend API Framework**: FastAPI, Uvicorn (`uvicorn[standard]`), Pydantic v2.
- **NLP & Telemetry Extraction**:
  - SpaCy (`en_core_web_sm` v3.x) for Named Entity Recognition (NER), POS tagging, and tokenization.
  - Sentence-Transformers (`all-MiniLM-L6-v2`) for sentence embeddings and cosine semantic drift.
  - NumPy, Pandas, Scikit-learn (for K-Means clustering and min-max scaling).
- **Neuro-Symbolic Reasoning**: PyReason (annotated interval logic and relational graph reasoning).
- **LLM Escalation Judge**: Google Generative AI SDK (`google-generativeai` / Gemini 1.5/2.0 API) + Mock Judge for offline test environments.
- **Frontend Layer**: React 19, Vite 8, Tailwind CSS v4, React Router v7.
- **Testing & Verification**: Pytest 9.x, AnyIO, HTTPX.

### 1.2 High-Level Component Topology
```
                  ┌──────────────────────────────────────────────┐
                  │                 Frontend UI                  │
                  │   (React 19 + Tailwind v4 + Vite Single SPA) │
                  └──────────────────────┬───────────────────────┘
                                         │ HTTP REST (Port 8000)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            FastAPI Backend Gateway           │
                  │          (backend.app.api.routes)            │
                  └──────┬───────────────┬───────────────┬───────┘
                         │               │               │
        ┌────────────────▼───┐  ┌────────▼────────┐  ┌───▼──────────────────┐
        │ Feature Extractor  │  │ Neuro-Symbolic  │  │ LLM Judge Escalation │
        │ (Default / Alt)    │  │ Gate (PyReason) │  │ (Gemini / Mock)      │
        └────────────────┬───┘  └────────┬────────┘  └───┬──────────────────┘
                         │               │               │
                         └───────────────┼───────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Fingerprint & Calibration Registry     │
                  │ (K-Means, T_L/T_H Grid Search, Thread Locks) │
                  └──────────────────────────────────────────────┘
```

---

## 2. Telemetry Extraction Specification (Stage 1)

Feature extraction maps arbitrary text $T$ into a 6-dimensional telemetry vector:
$$\mathbf{f}(T) = [H, S, C, E, D, M]^T \in \mathbb{R}^6$$

### 2.1 Feature Definitions & Mathematical Formulations

#### 1. $H$ — Hedge Density
Calculates the occurrence of qualifier phrases signaling low epistemic commitment ("might", "perhaps", "could be", "I think", "arguably", etc.):
$$H = \frac{\text{Count}(\text{hedge\_phrases} \in T)}{\text{Total Words}(T)} \times 100$$

#### 2. $S$ — Specificity
Measures factual granularity using SpaCy entity extraction and regular expressions for temporal/numerical tokens:
$$S = \frac{|\text{Named Entities}| + |\text{Numbers}| + |\text{Dates}|}{|\text{Total Tokens}|}$$

#### 3. $C$ — Citation Vagueness
Identifies authoritative gestures ("studies show", "according to research", "experts say") that are **not** immediately followed by a concrete named source within a window of $N=6$ tokens:
$$C = \frac{\text{Count}(\text{unanchored citation gestures})}{\max(1, \text{Total Sentences})}$$

#### 4. $E$ — Evidence Density
Estimates relational density by measuring sentences with concrete subject-predicate-object co-occurrences of named entities:
$$E = \frac{\sum_{s \in \text{Sentences}} \mathbb{I}(|\text{Entities}(s)| \ge 2)}{\max(1, |\text{Sentences}|)}$$

#### 5. $D$ — Semantic Drift
Measures semantic instability across sentences by computing the variance of sequential sentence embedding cosine similarities:
$$D = \operatorname{Var}\left(\left\{ \cos(\mathbf{e}_i, \mathbf{e}_{i+1}) \mid i = 1, \dots, K-1 \right\}\right)$$
where $\mathbf{e}_i = \text{SentenceTransformer}(s_i)$. If response length $K < 2$, $D = 0.0$.

#### 6. $M$ — Confidence Marker Density
Calculates the occurrence of absolute/emphatic assertions ("definitely", "absolutely", "guaranteed", "100%", "unquestionably"):
$$M = \frac{\text{Count}(\text{confidence\_phrases} \in T)}{\text{Total Words}(T)} \times 100$$

### 2.2 Per-Model Min-Max Normalization
Because base rates differ drastically across models (e.g., LLaMA-2 hedges more frequently than GPT-4), each extracted feature $f_k$ is normalized using the model's calibration parameters:
$$\hat{f}_k = \operatorname{clip}\left(\frac{f_k - \min_k}{\max_k - \min_k + \epsilon}, 0.0, 1.0\right)$$

---

## 3. Neuro-Symbolic Decision Gate Specification (Stage 2)

### 3.1 Symbolic Logic Graph (PyReason Representation)
Features are converted into interval facts $[L, U] \subseteq [0, 1]$:
```prolog
hedging_high(answer)     : [0.80, 1.00]
specificity_high(answer) : [0.75, 1.00]
citation_vague(answer)   : [0.85, 1.00]
evidence_low(answer)     : [0.70, 1.00]
confidence_high(answer)  : [0.80, 1.00]
drift_high(answer)       : [0.75, 1.00]
```

### 3.2 Anomaly Pattern Rules
Three relational rules model recognized hallucination archetypes:
```prolog
% Pattern 1: Evasive Fabrication (Hedges while gesturing to vague authorities)
anomaly_pattern_1(X) <- hedging_high(X), citation_vague(X)

% Pattern 2: Confident Fabrication (High surface specificity but lacks grounded evidence)
anomaly_pattern_2(X) <- specificity_high(X), evidence_low(X)

% Pattern 3: Unstable Overconfidence (Emphatic markers combined with topical/semantic drift)
anomaly_pattern_3(X) <- confidence_high(X), drift_high(X)
```

### 3.3 Gate Score Calculation & Verdict Logic
Let $r_i \in [0, 1]$ represent the activation strength of pattern $i$, and $w_i$ represent the calibrated rule weight ($\sum_{i=1}^3 w_i = 1$). The anomaly score $G$ is:
$$G = \sum_{i=1}^{3} w_i \cdot r_i$$

Decision boundary classification:
$$\text{Verdict}(G) = \begin{cases}
\text{LOW\_RISK}, & G < T_L \\
\text{AMBIGUOUS}, & T_L \le G \le T_H \\
\text{HIGH\_RISK}, & G > T_H
\end{cases}$$

---

## 4. Calibration Engine & Model Fingerprints

### 4.1 Fingerprint Data Structure
The model fingerprint encapsulates behavioral baselines and calibrated decision boundaries:
```json
{
  "model_id": "meta-llama/Llama-2-7b-chat-hf",
  "fingerprint_version": "v1.0.0",
  "feature_ranges": {
    "H": { "min": 0.0, "max": 4.5 },
    "S": { "min": 0.05, "max": 0.65 },
    "C": { "min": 0.0, "max": 2.0 },
    "E": { "min": 0.1, "max": 0.9 },
    "D": { "min": 0.0, "max": 0.15 },
    "M": { "min": 0.0, "max": 3.2 }
  },
  "centroids": {
    "factual": [0.21, 0.58, 0.05, 0.72, 0.02, 0.35],
    "hallucinated": [0.82, 0.31, 0.88, 0.22, 0.11, 0.79]
  },
  "thresholds": {
    "t_low": 0.35,
    "t_high": 0.68
  },
  "rule_weights": [0.40, 0.35, 0.25],
  "calibration_metadata": {
    "dataset": "HaluEval-QA",
    "sample_count": 1000,
    "calibrated_at": "2026-09-11T12:00:00Z"
  }
}
```

### 4.2 Threshold Optimization Algorithm
1. **Input**: Validation dataset $\mathcal{D}_{val} = \{(\mathbf{x}_j, y_j)\}_{j=1}^N$ where $y_j \in \{0, 1\}$.
2. **Objective Function**: Maximize classification accuracy on non-escalated cases while penalizing high escalation rates:
   $$\mathcal{L}(T_L, T_H, \mathbf{w}) = \operatorname{AUROC}_{\text{gate}} - \lambda \cdot \text{EscalationRate}(T_L, T_H)$$
   subject to $0.05 \le T_L < T_H \le 0.95$ and $T_H - T_L \ge 0.15$.
3. **Primary Dataset Enforcement**: Calibration logic executes `ensure_primary_only(records)` to filter out non-primary evaluation records.

---

## 5. Judge Escalation Pipeline (Stage 3)

### 5.1 Escalation Trigger
Invoked **strictly** when $T_L \le G \le T_H$. Responses scoring outside this interval bypass Stage 3 entirely.

### 5.2 Contextual Prompt Construction
The escalation prompt injects targeted symbolic telemetry:
```
You are a forensic LLM evaluation judge.
An automated neuro-symbolic gate evaluated the following answer to a factual question and found it AMBIGUOUS (score: {G}, threshold band: [{T_L}, {T_H}]).

Question: {question}
Answer: {answer}

The gate triggered the following anomaly patterns:
{triggered_patterns_json}

Model Baseline Behavioral Tells:
{fingerprint_summary}

Determine whether the answer contains factual hallucinations or fabrications.
Respond strictly in JSON format:
{
  "verdict": "LOW_RISK" | "HIGH_RISK",
  "confidence": float,
  "explanation": "concise forensic explanation of the verdict"
}
```

### 5.3 Fallback Mechanism
If the Gemini API key is missing or encounters rate-limiting (`ResourceExhausted`), the system engages `MockJudgePlugin` or returns standard HTTP 503 with informative diagnostic headers.

---

## 6. Plugin Architecture & Registry

### 6.1 Protocols
All extensible components implement PEP-544 runtime checkable protocols:
```python
@runtime_checkable
class FeatureExtractorPlugin(Protocol):
    name: str
    def extract(self, answer: str) -> dict[str, float]: ...

@runtime_checkable
class JudgePlugin(Protocol):
    name: str
    def judge(self, answer: str, fingerprint_summary: dict, ambiguous_patterns: list[dict]) -> JudgeResult: ...

@runtime_checkable
class DatasetLoaderPlugin(Protocol):
    name: str
    def load(self) -> Iterable[Record]: ...
```

### 6.2 Registry Operations
- Registries maintain thread-safe dictionaries: `_feature_extractors`, `_judges`, `_dataset_loaders`.
- Default registration happens at FastAPI startup (`lifespan` handler).

---

## 7. REST API Endpoints Specification

| Method | Endpoint | Description | Auth | Concurrency Guard |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/score` | Synchronously score an answer text | None (local) | Read-only |
| `GET` | `/api/v1/models` | List all calibrated model profiles | None (local) | Read-only |
| `GET` | `/api/v1/models/{id}/fingerprint` | Retrieve detailed model fingerprint | None (local) | Read-only |
| `POST` | `/api/v1/models/{id}/calibrate` | Trigger async calibration background job | None (local) | Mutex (`_calibration_lock`) |
| `GET` | `/api/v1/jobs/{id}` | Poll background calibration status | None (local) | Read-only |
| `GET` | `/api/v1/eval/summary` | Return benchmark evaluation metrics | None (local) | Read-only |
| `GET` | `/api/v1/health` | Health & uptime liveness probe | None (local) | Read-only |

### 7.1 Single-Port SPA Serving
FastAPI serves both API routes and static production assets:
```python
_dist_dir = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        target = _dist_dir / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(_dist_dir / "index.html")
```

---

## 8. Concurrency, Storage & Thread Safety

1. **State Persistence**:
   - In-memory registry for calibrated `Fingerprint` instances, enabling sub-millisecond retrieval.
   - File-based evaluation storage in `eval/results/*.json`.
2. **Thread Safety**:
   - `_calibration_lock = threading.Lock()` protects fingerprint dictionary updates.
   - `_active_calibrations = set()` guarantees that only one calibration job can execute per model ID at any given time.

---

## 9. Verification & Testing Strategy

- **Test Framework**: Pytest with 121 unit tests across 9 comprehensive suites:
  - `test_feature_extraction.py`: Validates mathematical boundaries of H, S, C, E, D, M.
  - `test_gate.py`: Validates PyReason pattern activation and scoring bounds.
  - `test_adaptive_gate.py`: Validates grid-search threshold convergence and edge-case fallbacks.
  - `test_judge.py`: Validates Gemini prompt formatting and Mock Judge failover.
  - `test_api.py`: Validates all 7 FastAPI endpoints against schema contracts.
  - `test_fingerprint.py`: Validates K-Means clustering and primary dataset isolation.
  - `test_data_ingestion.py`: Validates HaluEval, TruthfulQA, and SimpleQA loaders.
  - `test_eval_harness.py`: Validates AUROC, AUPRC, and ECE computation algorithms.
  - `test_full_eval.py`: Validates complete end-to-end evaluation pipeline.
