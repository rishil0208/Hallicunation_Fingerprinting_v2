# Hallucination Fingerprinting Gate

**A per-model adaptive hallucination detection system** that builds unique "fingerprints" of how different language models hallucinate, using symbolic pattern rules and an LLM judge escalation pipeline.

> **Research Prototype.** This is a research demonstration, not a production fact-checking system. Verdicts reflect statistical pattern analysis on calibration data and should not be treated as ground truth.

## Architecture

```
Answer Text
    ↓
┌─────────────────────────┐
│ Stage 1: Feature        │  6 telemetry features:
│ Extraction              │  H(edge), S(pecificity), C(itation),
│                         │  E(vidence), D(rift), M(confidence)
└─────────┬───────────────┘
          ↓
┌─────────────────────────┐
│ Stage 2: Symbolic Gate  │  G = Σ w_i × r_i
│ (per-model thresholds)  │  3 anomaly patterns → classify
│                         │  LOW_RISK / AMBIGUOUS / HIGH_RISK
└─────────┬───────────────┘
          ↓ (AMBIGUOUS only)
┌─────────────────────────┐
│ Stage 3: LLM Judge      │  Gemini API call with anonymized
│ (escalation)            │  prompt → RESOLVED_AMBIGUOUS
└─────────────────────────┘
```

## Core Contribution

Per-model adaptive thresholds `(T_L, T_H, w_i)` vs. global pooled thresholds — the system learns that different models hallucinate differently and calibrates detection parameters accordingly.

## Quick Start

### Backend

```bash
# Create and activate Python 3.10 virtualenv
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run tests (121 tests)
python -m pytest backend/tests/ -v

# Start API server
uvicorn backend.app.api.routes:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies API calls to `http://localhost:8000`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/score` | Score an answer for hallucination risk |
| `GET` | `/api/v1/models` | List calibrated models |
| `GET` | `/api/v1/models/{id}/fingerprint` | Get model fingerprint |
| `POST` | `/api/v1/models/{id}/calibrate` | Trigger calibration job |
| `GET` | `/api/v1/jobs/{id}` | Check job status |
| `GET` | `/api/v1/eval/summary` | Evaluation results |
| `GET` | `/api/v1/health` | Liveness check |

## Project Structure

```
AI_PROJECT/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── schemas.py             # Pydantic models
│   │   ├── registry.py            # Plugin registry
│   │   ├── fingerprint/
│   │   │   └── cluster.py         # k-means + label alignment
│   │   ├── gate/
│   │   │   ├── rules.py           # Symbolic rule definitions
│   │   │   ├── score.py           # Gate scoring (pure)
│   │   │   ├── calibrate.py       # Threshold calibration
│   │   │   └── escalate.py        # Judge escalation
│   │   └── plugins/
│   │       ├── datasets/          # HaluEval, TruthfulQA, SimpleQA
│   │       ├── features/          # Default + Alt extractors
│   │       └── judges/            # Mock + Gemini judges
│   └── tests/                     # 121 unit tests
├── eval/
│   ├── metrics.py                 # AUROC, AUPRC, ECE
│   ├── baselines.py               # All 5 spec baselines
│   └── run_experiment.py          # Evaluation harness
├── frontend/                      # React + Tailwind
│   └── src/
│       ├── components/
│       │   ├── FingerprintRadar.jsx
│       │   └── Disclaimer.jsx
│       └── pages/
│           ├── ScorePage.jsx
│           ├── ModelsPage.jsx
│           └── EvalPage.jsx
├── data/raw/                      # HaluEval QA dataset
├── configs/default.yaml
└── docs/
    ├── decisions.md               # ADRs 1-8
    └── Hallucination_Fingerprinting_Master_Spec_v2.md
```

## Design System: Signal Forensics

- **Graphite ink** background (#0B0D10, #14171C, #262B33)
- **Paper** text (#E8EAED primary, #8B93A1 muted)
- **Three semantic colors**: Teal (LOW_RISK), Amber (AMBIGUOUS), Coral (HIGH_RISK)
- **Typography**: Space Grotesk (display), Inter (body), JetBrains Mono (data)

## Key Decisions (ADRs)

| # | Decision | Rationale |
|---|----------|-----------|
| A-C1 | Judge failure → HTTP 502 | Never fake RESOLVED_AMBIGUOUS |
| A-C2 | Post-hoc cluster label alignment | k-means assigns arbitrary IDs |
| A-H1 | Explicit None-check before classify | Prevent TypeError on uncalibrated models |
| A-H2 | Sanitize Gemini exceptions | API keys must never leak |
| A-M1 | Anonymize model identity in judge prompts | Prevent identity bias |
| A-M2 | Constrain 0 ≤ T_L < T_H ≤ 1 | Ensure valid threshold bands |

## Reproducibility

- Fixed random seeds throughout (42 default)
- Versioned data splits persisted to `data/processed/splits/`
- Git commit hash tracked per experiment run
- `setuptools<81` pinned for PyReason compatibility

## Known Limitations

- Single model_id (chatgpt) in HaluEval — per-model comparison requires simulation
- In-memory job store (no persistence across restarts)
- Single-user research demo, not production-scaled

## License

Research prototype — TAM (The AI and ML Club), VIT Vellore.
