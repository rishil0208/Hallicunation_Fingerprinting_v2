<div align="center">

# 🧠 Hallucination Fingerprinting Gate (HFG)
### *A Neuro-Symbolic, Model-Agnostic Framework for Calibrated Hallucination Detection*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/Tests-128%20Passed%20(100%25)-success?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<br />

**HFG** is an explainable, neuro-symbolic hallucination detection framework. Rather than treating all LLMs identically with rigid global thresholds, HFG profiles each target model’s latent behavioral characteristics into a distinct **"Fingerprint"** and verifies responses through a cascaded 3-stage neuro-symbolic pipeline.

[Architecture](#-architecture) • [Key Features](#-key-features) • [Telemetry Features](#-the-6-telemetry-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Developer](#-developer)

---

</div>

## 📌 The Problem & Innovation

Current hallucination detection methods suffer from two fatal flaws:
1. **Model Blindness:** An answer generated with moderate entropy might be routine for a large model (e.g. LLaMA-3-70B) but a severe anomaly for a compact model. Universal detection thresholds produce high false-positive rates.
2. **The "LLM-as-a-Judge" Cost Trap:** Evaluating every single generated token using another expensive LLM (like GPT-4 or Claude 3.5) causes massive API latency and prohibitive costs.

### ✨ The HFG Solution
* **Per-Model Calibration:** HFG learns distinct baseline clusters and adaptive thresholds $(T_L, T_H, w_i)$ for each model (LLaMA-3, Mistral, GPT-4, etc.).
* **Neuro-Symbolic Efficiency:** Over 85% of queries are resolved instantly via deterministic mathematical telemetry and symbolic logic gates, invoking an LLM judge only for ambiguous edge cases.
* **100% Explainability:** Violations are attributed directly to activated symbolic anomaly rules rather than a black-box probability.

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────┐
                    │   Input Prompt & AI Answer   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │       STAGE 1: Mathematical Telemetry Extraction      │
        │   Extracts 6 features: H, S, C, E, D, M via spaCy,   │
        │           Sentence-Transformers & Regex              │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │          STAGE 2: Symbolic PyReason Logic Gate       │
        │  Evaluates features against Model-Specific Baselines │
        │              Gate Score: G = Σ (w_i × r_i)           │
        └──────────────┬───────────────────────┬───────────────┘
                       │                       │
         [G < T_L]     │         [G > T_H]     │  [T_L ≤ G ≤ T_H]
        ┌──────────────┘                       └──────────────┐
        ▼                                                     ▼
┌───────────────┐                             ┌───────────────────────────────┐
│   LOW RISK    │                             │      STAGE 3: LLM Judge       │
│ (Symbolic OK) │                             │   (Escalation Tie-Breaker)    │
└───────────────┘                             │ Evaluates Ambiguous Edge Case │
        ▲                                     └──────────────┬────────────────┘
        │               [Judge Rulings]                      │
        └────────────────────────────────────────────────────┘
```

---

## 📊 The 6 Telemetry Features

HFG extracts six orthogonal signals that distinguish truthful completions from hallucinations:

| Symbol | Feature | Description | Extraction Method |
| :---: | :--- | :--- | :--- |
| **H** | **Hedge Density** | Frequency of linguistic hedges (*maybe, perhaps, arguably, possibly*) | Lexical pattern density |
| **S** | **Specificity** | Proportion of concrete named entities, dates, numbers, and technical identifiers | spaCy NER + token regex |
| **C** | **Citation Vagueness** | Unverified source gestures (*studies show, sources indicate*) without named anchors | Distance-window NER parsing |
| **E** | **Evidence Density** | Verifiable factual assertions and relation triples per sentence | POS tagger + dependency parse |
| **D** | **Semantic Drift** | Cosine-similarity variance across successive sentence embeddings | MiniLM-L6-v2 pairwise variance |
| **M** | **Confidence Density** | Over-confidence assertions (*definitely, 100%, unquestionably*) | Lexicon occurrence rate |

---

## 🔬 Neuro-Symbolic Logic Rules

The Symbolic Gate synthesizes normalized telemetry vectors into anomaly activations:
* **Anomaly Pattern 1 (`H` + `C`):** High hedging paired with vague, anonymous citations.
* **Anomaly Pattern 2 (`S` + `E`):** High surface specificity (dates/names) with low checkable evidence (fabricated details).
* **Anomaly Pattern 3 (`M` + `D`):** High dogmatic confidence accompanied by drastic semantic topic drift.

---

## 🚀 Quick Start

### Prerequisites
* **Python 3.10+**
* **Node.js 18+ & npm** (for frontend)
* **Ollama (Optional but Recommended):** For 100% offline, privacy-preserving LLM Judge escalation.

### 1. One-Click Launch (Recommended)

#### 🪟 Windows:
Double-click `run.bat` or run:
```cmd
run.bat
```

#### 🐧 Linux / macOS:
```bash
chmod +x start_linux.sh
./start_linux.sh
```

This single command starts both the **FastAPI Backend (port 8000)** and the **React Dashboard (port 5173 / port 8000)**.

---

### 2. Run with Local Qwen 2.5 Judge (Ollama)

HFG can escalate ambiguous cases to a local Qwen 2.5 judge, processing structured JSON telemetry completely offline.

1. **Start Ollama and pull the model:**
   ```bash
   ollama serve
   ollama pull qwen2.5:7b
   ```
2. **Start the backend:** (It automatically detects and uses Qwen when available)
   ```bash
   ./start_linux.sh
   ```

---

### 3. Manual Setup

```bash
# Clone the repository
git clone https://github.com/rishil0208/Hallicunation_Fingerprinting_v2.git
cd Hallicunation_Fingerprinting_v2

# 1. Setup Python Virtual Environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
python -m spacy download en_core_web_sm

# 2. Run Test Suite
pytest backend/tests/ -v

# 3. Start Backend
uvicorn backend.app.api.routes:app --host 127.0.0.1 --port 8000

# 4. Start Frontend (New Terminal)
cd frontend
npm install
npm run dev
```

Visit **[http://localhost:8000](http://localhost:8000)** or **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/score` | Submit prompt/answer and receive 3-stage verdict & telemetry |
| `GET` | `/api/v1/models` | List all calibrated model profiles & fingerprint versions |
| `GET` | `/api/v1/models/{id}/fingerprint` | Retrieve radar centroids, normalization mins/maxs, and thresholds |
| `POST` | `/api/v1/models/{id}/calibrate` | Trigger background K-Means clustering and threshold sweep |
| `GET` | `/api/v1/health` | Service health and liveness check |

#### Example Request (`POST /api/v1/score`)
```json
{
  "model_id": "gpt-3.5-turbo",
  "answer": "Studies show that Neil Armstrong might have arguably discovered an ancient pyramid base on Mars, but sources indicate it is not certain."
}
```

#### Example Response
```json
{
  "verdict": "HIGH_RISK",
  "gate_score": 0.3600,
  "thresholds": {
    "t_low": 0.0800,
    "t_high": 0.2200
  },
  "resolved_by": "gate",
  "triggered_patterns": [
    {
      "name": "anomaly_pattern_1",
      "features_involved": ["H", "C"],
      "strength": 1.0000
    }
  ],
  "feature_breakdown": {
    "H": 1.0000,
    "S": 0.0000,
    "C": 1.0000,
    "E": 0.0000,
    "D": 0.0108,
    "M": 0.0000
  },
  "explanation": "High hallucination risk: triggered [anomaly_pattern_1] (H, C).",
  "explanation_source": "symbolic",
  "model_id": "gpt-3.5-turbo",
  "fingerprint_version": "v1"
}
```

---

## 📂 Project Structure

```
Hallicunation_Fingerprinting_v2/
├── backend/
│   ├── app/
│   │   ├── api/routes.py            # FastAPI REST endpoints & SPA hosting
│   │   ├── fingerprint/cluster.py   # K-Means clustering & label alignment
│   │   ├── gate/                    # Neuro-symbolic gate, rules, & scoring
│   │   ├── plugins/                 # Extensible feature extractors & LLM judges
│   │   ├── registry.py              # Dynamic plugin registry
│   │   └── schemas.py               # Pydantic data models
│   └── tests/                       # 128 unit tests (100% pass)
├── frontend/
│   ├── src/
│   │   ├── pages/ScorePage.jsx      # Interactive scoring & live presets
│   │   ├── pages/ModelsPage.jsx     # Fingerprint Radar visualizer
│   │   ├── components/              # Reusable UI components
│   │   └── api.js                   # Backend API client
│   └── dist/                        # Pre-compiled production bundle
├── data/
│   └── processed/fingerprints/      # Persisted model baselines (JSON)
├── run.bat                          # One-click Windows runner
├── start_linux.sh                   # Unified Linux runner
├── PRD.md                           # Product Requirements Document
├── TRD.md                           # Technical Requirements Document
└── DATAFLOW.md                      # Dataflow & Sequence Diagrams
```

---

## 🧪 Test Suite & Verification

All core modules are covered by **128 automated unit tests**:
```bash
pytest backend/tests/ -v
# ======================== 128 passed in ~7s ========================
```
* **Adaptive Gate Tests:** Threshold constraint proofs, fallback validation, boundary verification.
* **Clustering & Alignment:** K-Means label alignment and normalization invariance.
* **API Endpoints:** Request validation, health checks, error handling, and payload formatting.

---

## 👨‍💻 Developer

Developed by **Rishil** ([@rishil0208](https://github.com/rishil0208))  
*Specialized research in AI Evaluation, Neuro-Symbolic Systems, and Trustworthy LLM Architecture.*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
