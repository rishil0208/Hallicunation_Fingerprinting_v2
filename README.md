# Hallucination Gate

Model-Calibrated Neuro-Symbolic Decision Gate for Hallucination Detection.

A research prototype for TAM (The AI and ML Club, VIT Vellore).

## Overview

Three-stage pipeline:
1. **Feature Extraction** — six deterministic linguistic features (H, S, C, E, D, M)
2. **Symbolic Gate** — PyReason rule graph with per-model calibrated thresholds
3. **LLM Judge** — Gemini escalation for ambiguous verdicts only

## Setup

```bash
python3 -m venv --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Running

```bash
# Start the API server
uvicorn backend.app.main:app --reload

# Run tests
pytest
```

## Project Structure

See `docs/Hallucination_Fingerprinting_Master_Spec_v2.md` Section 7 for the full layout.

## Disclaimer

This is a research prototype, not a production fact-checking system.
