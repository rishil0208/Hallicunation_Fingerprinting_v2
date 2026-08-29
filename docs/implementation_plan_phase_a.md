# FINAL DISCOVERY REPORT — Hallucination Fingerprinting System

**Date**: 2026-08-29  
**Phase**: A — Final Discovery (Reconciled)  
**Role**: Principal Architect  
**Repository**: `/home/rishil-n/AI_PROJECT`  
**Authoritative Spec Source**: [Claude Text v2.md](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md) (815 lines, 61,613 bytes)

---

## Audit Reconciliation Summary

The independent audit raised **3 critical**, **3 high-priority**, and **3 medium** findings. After verification against actual files and system state, here is the disposition:

| # | Audit Finding | Disposition | Reasoning |
|---|---|---|---|
| C1 | No Java/JRE for PyReason | ❌ **REJECTED** | PyReason is pure Python, built on Numba for JIT compilation. It does **not** use Java or JPype. Verified via [web search against PyReason's `setup.py`](https://pypi.org/project/pyreason/). Dependencies are `networkx`, `pyyaml`, `pandas`, `numba`, `numpy`, `memory_profiler`, `pytest`. |
| C2 | No `pip` / `pip3` | ✅ **ACCEPTED** | Verified: `python3 -m pip --version` → `No module named pip`; `pip3` → `command not found`; `python3 -m ensurepip` → `No module named ensurepip`. No `conda`/`poetry`/`pipenv`/`pyenv` present. |
| C3 | No datasets on filesystem | ✅ **ACCEPTED** | Verified: `find /home/rishil-n -iname "*halueval*" -o -iname "*qa_data*" -o -iname "*truthful*" -o -iname "*simpleqa*"` returned zero results. |
| H1 | Omitted graphite/paper hex values | ✅ **ACCEPTED** | Verified against spec [Section 5.13, lines 259–266](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L259-L266). The original discovery report listed only 3 accent colors and omitted `#0B0D10`, `#14171C`, `#262B33`, `#E8EAED`, `#8B93A1`. |
| H2 | API route path param names wrong | ✅ **ACCEPTED** | Verified against spec [Section 5.12, lines 199–208](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L199-L208). Spec uses `{model_id}` and `{job_id}`, not `{id}`. |
| H3 | Data leakage risk | ✅ **ACCEPTED** | Spec [Section 5.6, line 135](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L135) explicitly states thresholds are "learned on a validation split." Train/val/test must be strictly separated per model. |
| M1 | Python 3.14 compatibility risk | ✅ **ACCEPTED, UPGRADED to HIGH** | PyReason's PyPI metadata restricts support to Python 3.9–3.10 for parallel Numba features. Python 3.14 is not in its supported versions list. This is now the primary PyReason blocker (replacing the rejected Java finding). |
| M2 | `RESOLVED_AMBIGUOUS` verdict ambiguity | ⚠️ **PARTIALLY REJECTED** | The spec is internally consistent. Section 5.6 defines three internal states; Section 5.12 maps the post-judge result to `RESOLVED_AMBIGUOUS` (not back to `LOW_RISK`/`HIGH_RISK`), which is an intentional transparency signal aligned with the honesty principle in Section 5.8. However, the spec does not define how the consumer learns the *direction* of the resolution (low vs. high) — this gap is real and should be clarified. |
| M3 | PyReason bounds → `r_i` undefined | ✅ **ACCEPTED** | Spec [Section 5.5, line 131](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L131) defines `G = Σ w_i * r_i` but never specifies how `r_i` is derived from the `[lower, upper]` interval. |
| — | Spec source file found outside workspace | ✅ **ACCEPTED** | Verified: [Claude Text v2.md](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md) is 61,613 bytes, SHA-256 `6d6926bb...`. The VS Code backup at `/home/rishil-n/.config/Code/Backups/.../file/-36afde8d` has a different hash (`b3c81d2a...`) and contains VS Code metadata on line 1 — it is a backup copy with metadata prepended, not an independent version. |

---

## 1. Confirmed Repository State

Verified via `ls -laR /home/rishil-n/AI_PROJECT/` and `find ... -exec wc -c`:

```
AI_PROJECT/                           (no .git)
└── docs/
    └── Hallucination_Fingerprinting_Master_Spec_v2.md   (0 bytes)
```

| Attribute | Status | Evidence |
|---|---|---|
| Git initialized | ❌ No | `git status` → `fatal: not a git repository` |
| Spec file | ❌ Empty (0 bytes) | `wc -c` → `0` |
| Source code | ❌ None | `find -type f` returns only the spec |
| Tests | ❌ None | No `tests/` directory |
| Config files | ❌ None | No `pyproject.toml`, `package.json`, etc. |
| Data directories | ❌ None | No `data/` directory |
| `docs/decisions.md` | ❌ Missing | Only `docs/Hallucination_Fingerprinting_Master_Spec_v2.md` exists |
| Datasets on disk | ❌ None anywhere in `/home/rishil-n/` | `find -iname` returned empty |

> [!IMPORTANT]
> The **full specification text** exists at [/home/rishil-n/Documents/Interview Perp/Claude Text v2.md](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md) (815 lines, 61,613 bytes). Phase B's first action must copy this content into the workspace's `docs/Hallucination_Fingerprinting_Master_Spec_v2.md`.

---

## 2. Confirmed Architecture (Spec vs. Repository)

### What the Spec Defines (verified against [Section 5, lines 80–304](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L80-L304))

A three-stage pipeline wrapped in an API and frontend:

| Stage | Component | Spec Section |
|---|---|---|
| Data | `Record{question, answer, model_id, label, source_dataset, dataset_role}` | [5.1, L82–87](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L82-L87) |
| 0 | Six features H, S, C, E, D, M (deterministic, no LLM) | [5.3, L93–106](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L93-L106) |
| 1 | Per-model fingerprint clustering (k-means/GMM) | [5.4, L108–110](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L108-L110) |
| 2 | PyReason symbolic rule graph → `G = Σ w_i * r_i`, per-model `(T_L, T_H, w_i)` | [5.5–5.6, L112–142](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L112-L142) |
| 3 | LLM judge (Gemini) escalation, ONLY when `T_L ≤ G ≤ T_H` | [5.7, L144–147](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L144-L147) |
| API | FastAPI with 7 endpoints under `/api/v1/*` | [5.12, L167–217](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L167-L217) |
| UI | React + Tailwind, "Signal Forensics" design, Fingerprint Radar | [5.13, L249–298](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L249-L298) |
| Plugins | `FeatureExtractorPlugin`, `JudgePlugin`, `DatasetLoaderPlugin` | [5.12, L219–243](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L219-L243) |

### What Actually Exists

**Nothing.** Zero implementation. Zero infrastructure.

### Spec's Own Repo Structure (Section 7, [lines 347–409](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L347-L409))

The spec prescribes the root directory be named `hallucination-gate/` with `backend/` and `frontend/` at the top level — **not** a `src/` layout. The original discovery report proposed a `src/hallucination_fingerprinting/` layout which diverges from the spec. Phase B must follow the spec's structure.

---

## 3. Confirmed Existing Components

None. Zero reusable code, configuration, or data.

---

## 4. Confirmed Missing Components

Every component in the spec is missing. The complete list (grouped by spec section):

**Data Layer** ([5.1–5.2](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L82-L91)):
- `Record` model with `dataset_role ∈ {"primary", "secondary"}`
- HaluEval QA loader (`qa_data.json`, primary), TruthfulQA loader (secondary), SimpleQA loader (secondary)
- `dataset_role` filter enforced in calibration code, with test asserting it

**Feature Extraction** ([5.3](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L93-L106)):
- H (hedge density), S (specificity), C (citation vagueness), E (evidence density), D (semantic/entity drift), M (confidence-marker density)
- Per-model `[0,1]` normalization using training-set distribution
- Default `FeatureExtractorPlugin` + one alternate implementation (spec [line 240](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L240))

**Fingerprinting** ([5.4](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L108-L110)):
- Per-model k-means/GMM clustering, versioned serializable fingerprint file

**Symbolic Gate** ([5.5–5.6](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L112-L142)):
- PyReason rule graph with annotated interval bounds
- `G = Σ w_i * r_i`, per-model `(T_L, T_H)` and `w_i`
- Calibration pipeline (grid/Bayesian search on validation split)

**LLM Judge** ([5.7](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L144-L147)):
- `GeminiJudgePlugin`, `MockJudgePlugin` (test/local default)
- Escalation logic: invoke ONLY when `T_L ≤ G ≤ T_H`

**Backend** ([5.12](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L167-L246)):
- FastAPI app with these exact routes:
  - `POST /api/v1/score`
  - `GET  /api/v1/models`
  - `GET  /api/v1/models/{model_id}/fingerprint`
  - `POST /api/v1/models/{model_id}/calibrate`
  - `GET  /api/v1/jobs/{job_id}`
  - `GET  /api/v1/eval/summary`
  - `GET  /api/v1/health`
- Plugin registry (dict-based, config-driven)
- Background job system for calibration

**Frontend** ([5.13](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L249-L298)):
- React + Tailwind with exact design tokens (8 colors, 3 typefaces)
- Pages: Score a Response, Model Fingerprints, Evaluation, Disclaimer
- Fingerprint Radar component (reusable)
- `VerdictBand`, `FeatureBar`, `Disclaimer` components

**Evaluation** ([Section 6](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L308-L340)):
- 5 baselines (always-escalate, never-escalate, global-threshold, random, TRACT-style)
- Core experiment: adaptive vs. global on ≥2–3 models
- Metrics: AUROC, AUPRC, ECE, escalation rate, latency, cost

**Infrastructure**:
- Git, `.gitignore`, `docs/decisions.md`, README, configs/, data/raw/ + data/processed/

---

## 5. Confirmed Broken Components

Only one file exists, and it is broken:

| File | Issue |
|---|---|
| [Hallucination_Fingerprinting_Master_Spec_v2.md](file:///home/rishil-n/AI_PROJECT/docs/Hallucination_Fingerprinting_Master_Spec_v2.md) | 0 bytes. The full spec content (61,613 bytes) exists at [Claude Text v2.md](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md) and must be copied in. |

---

## 6. Confirmed Dependencies

### System-Level (Verified)

| Dependency | Status | Evidence |
|---|---|---|
| Python 3.14.4 | ✅ Present | `python3 --version` → `Python 3.14.4` |
| `pip` / `pip3` | ❌ **Missing** | `python3 -m pip` → `No module named pip`; `ensurepip` also missing |
| Node.js 22.22.1 | ✅ Present | `node --version` → `v22.22.1` |
| npm 9.2.0 | ✅ Present | `npm --version` → `9.2.0` |
| Java / JRE | ⬜ **Not needed** | PyReason does not require Java (uses Numba) |
| `apt` | ✅ Present | `apt-get --version` → `apt 3.2.0` — can install `python3-pip` |

### Python Packages (All Missing — cannot install without pip)

| Package | Purpose | Spec Reference |
|---|---|---|
| `pyreason` | Symbolic rule graph (Stage 2) | [5.5, L112](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L112) |
| `numba` | PyReason transitive dependency | PyReason's setup.py |
| `networkx` | PyReason transitive dependency | PyReason's setup.py |
| `fastapi` + `uvicorn` | Backend API | [5.12, L167](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L167) |
| `pydantic` | Data models, API schemas | [5.12, L174](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L174) |
| `scikit-learn` | Clustering (Stage 1), calibration | [5.4, L110](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L110) |
| `spacy` | NER for feature S | [5.3, L100](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L100) |
| `sentence-transformers` | Embeddings for feature D | [5.3, L103](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L103) |
| `google-generativeai` | Gemini LLM judge | [5.7, L144](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L144) |
| `numpy`, `pandas` | Numerical computation | PyReason deps + general |
| `pytest`, `httpx` | Testing | Standard |

### Frontend Packages (All Missing)

| Package | Purpose |
|---|---|
| `react`, `react-dom` | UI framework |
| `vite` | Build tool |
| `tailwindcss` | Utility CSS |
| `@fontsource/space-grotesk`, `@fontsource/inter`, `@fontsource/jetbrains-mono` | Typography |
| Charting library (recharts/d3/custom SVG) | Fingerprint Radar |

---

## 7. Confirmed Blockers (ordered by severity)

> [!CAUTION]
> These must be resolved before Phase B implementation can begin.

### Blocker 1 — No `pip` (CRITICAL)

`python3 -m pip` → `No module named pip`. No `ensurepip`, no `conda`, no `poetry`. Cannot install any Python packages.

**Resolution**: Run `sudo apt-get install python3-pip` (or `python3-venv` and bootstrap from there). Requires user action.

### Blocker 2 — Python 3.14 + PyReason incompatibility (HIGH)

PyReason's PyPI metadata restricts support to Python 3.9–3.10. Numba itself supports 3.14 (since v0.63.0), but PyReason has not updated its classifier. Installing PyReason on 3.14 may fail or produce runtime errors in Numba's JIT compilation.

**Resolution options** (require human decision):
- A) Install Python 3.10 via `apt` or `pyenv`, use a venv
- B) Attempt PyReason on 3.14, fall back to (A) if it fails
- C) Accept the risk and pin to latest PyReason, testing immediately

### Blocker 3 — Empty spec file in workspace (HIGH)

The workspace file `docs/Hallucination_Fingerprinting_Master_Spec_v2.md` is 0 bytes. The authoritative content is at `/home/rishil-n/Documents/Interview Perp/Claude Text v2.md`.

**Resolution**: Copy the content into the workspace file as Phase B's first step.

### Blocker 4 — No datasets on disk (MEDIUM)

HaluEval QA, TruthfulQA, and SimpleQA are not present. Downloading requires network access (sandbox bypass).

**Resolution**: Download via bypass-sandbox commands, or have the user place files manually.

---

## 8. Confirmed Research Risks

| # | Risk | Severity | Evidence |
|---|---|---|---|
| R1 | **Per-model data sufficiency** | 🔴 High | HaluEval QA has ~10k samples ([spec L324](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L324)), but each model's share may be small. Per-model `(T_L, T_H, w_i)` calibration with few hundred samples per model risks overfitting. Spec [Section 6, L337](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L337) acknowledges this via "sensitivity analysis" but does not define a minimum sample threshold. |
| R2 | **PyReason bounds → G semantics** | 🟡 Medium | Spec [line 131](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L131) defines `G = Σ w_i * r_i` but never specifies how `r_i` (rule activation strength) is extracted from PyReason's `[lower, upper]` interval bounds. Must be defined at implementation time and documented in `decisions.md`. |
| R3 | **Feature overlap: H vs. M** | 🟡 Medium | H (hedge density) and M (confidence-marker density) are both lexicon-based word-count features. Spec [line 335](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L335) calls for "feature-subset ablation" which would surface this, but correlation analysis should happen early. |
| R4 | **RESOLVED_AMBIGUOUS direction gap** | 🟡 Medium | Spec [5.12, L183](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L183) returns `RESOLVED_AMBIGUOUS` as the verdict after judge escalation, but does not include a sub-field indicating whether the judge resolved it toward low-risk or high-risk. The `explanation` field carries this in natural language, but no structured field exists for it. |
| R5 | **LLM judge non-determinism** | 🟡 Medium | Spec [5.7](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L144) does not specify `temperature=0`, seed, or prompt versioning for Gemini calls. Without these, evaluation results will vary across runs. |
| R6 | **TruthfulQA label mismatch** | 🟡 Low | Spec [5.2, L91](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L91) explicitly acknowledges TruthfulQA labels reflect factuality, not fabrication. This is already mitigated by `dataset_role = "secondary"`, but must be documented in any paper. |

---

## 9. Confirmed Engineering Risks

| # | Risk | Severity | Details |
|---|---|---|---|
| E1 | **No VCS** | 🔴 High | No `.git`. Must init immediately in Phase B. |
| E2 | **Greenfield scope** | 🔴 High | Full-stack: 6 NLP features + PyReason + calibration + FastAPI (7 endpoints) + React frontend (4 pages + Radar component) + plugin system + evaluation pipeline. |
| E3 | **PyReason on Python 3.14** | 🔴 High | See Blocker 2. May require Python version downgrade. |
| E4 | **API key management** | 🟡 Medium | Gemini API key needed for `GeminiJudgePlugin`. Spec [L453](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L453) mandates it is server-side only, never exposed to frontend. `MockJudgePlugin` is the safe default. |
| E5 | **Strict implementation sequence** | 🟡 Medium | Spec [Section 7, L412–424](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L412-L424) mandates a strict 11-step build order. Pipeline must be real before API wraps it, API must be real before frontend calls it. No stubs. |

---

## 10. Confirmed Specification Mismatches (Original Discovery Report vs. Spec)

| # | Issue | Spec Citation | Correction Required |
|---|---|---|---|
| M1 | **Repo layout**: Discovery proposed `src/hallucination_fingerprinting/`. Spec prescribes `hallucination-gate/backend/app/` | [Section 7, L347–409](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L347-L409) | Phase B must follow the spec's `hallucination-gate/` structure |
| M2 | **Route params**: Discovery used `{id}` for models and jobs routes | [Section 5.12, L199–208](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L199-L208) | Must be `{model_id}` and `{job_id}` |
| M3 | **Design tokens incomplete**: Discovery listed only 3 accent hex codes | [Section 5.13, L259–266](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L259-L266) | Full palette is 8 colors: `#0B0D10`, `#14171C`, `#262B33`, `#E8EAED`, `#8B93A1`, `#4FE3C1`, `#F2B84B`, `#FF6B4A` |
| M4 | **Missing alt_extractor.py**: Discovery's file tree omits the required second feature-extractor plugin | [Section 5.12, L240](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L240) | Must build at least one alternate implementation |
| M5 | **Missing frontend components**: Discovery's file tree omits `VerdictBand.tsx`, `FeatureBar.tsx`, `Disclaimer.tsx`, `tokens.ts` | [Section 7, L387–394](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L387-L394) | Must include all four |
| M6 | **Missing eval/ directory**: Discovery omits `eval/metrics.py`, `eval/baselines.py`, `eval/run_experiment.py` | [Section 7, L401–404](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L401-L404) | Must include evaluation pipeline at repo root |
| M7 | **Missing gate/ directory**: Discovery omits `backend/app/gate/rules.py`, `graph.py`, `score.py` | [Section 7, L373–376](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L373-L376) | Must include gate module |

---

## 11. Exact Scope for Phase B

### Phase B Prerequisite Actions (before any code)

1. **Copy spec content** from `/home/rishil-n/Documents/Interview Perp/Claude Text v2.md` into `docs/Hallucination_Fingerprinting_Master_Spec_v2.md`
2. **Install `pip`** — requires `sudo apt-get install python3-pip` (or `python3-venv` + bootstrap)
3. **Resolve Python version** — test PyReason on 3.14; if incompatible, install Python 3.10 via `apt` and create a venv
4. **Initialize git** — `git init`, create `.gitignore`, initial commit
5. **Download datasets** — HaluEval QA (`qa_data.json`), TruthfulQA, SimpleQA into `data/raw/`

### Phase B Implementation (following spec's strict order, [Section 7, L412–424](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L412-L424))

**Step 1**: Data ingestion + normalization (HaluEval QA first)
- `Record` model with `dataset_role`
- HaluEval loader → `dataset_role = "primary"`
- Test asserting `dataset_role` filter works
- Then TruthfulQA/SimpleQA loaders → `dataset_role = "secondary"`

**Step 2**: Feature extraction (H, S, C, E, D, M)
- Default `FeatureExtractorPlugin` + one alternate
- Unit tests on hand-crafted sentences

**Step 3**: Baselines 1 (always-escalate) and 4 (random/majority-class)
- Evaluation harness end-to-end

**Step 4**: Fingerprint clustering (Stage 1)

**Step 5**: Symbolic gate with global fixed threshold (Baseline 3)

**Step 6**: Per-model adaptive calibration (core contribution)

**Step 7**: LLM judge (`MockJudgePlugin` first, `GeminiJudgePlugin` second)

**Step 8**: Full evaluation + ablations

**Step 9**: FastAPI backend (after pipeline is real)

**Step 10**: Frontend (after API is real)

**Step 11**: Documentation

---

## 12. Files Phase B Should Create

Following the spec's prescribed structure ([Section 7, L347–409](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L347-L409)):

```
hallucination-gate/                          ← spec's prescribed root name
  backend/
    app/
      main.py                                ← FastAPI entrypoint
      api/
        routes_score.py
        routes_models.py
        routes_jobs.py
        routes_eval.py
      schemas.py                             ← Pydantic models (Section 5.12 contract)
      plugins/
        registry.py
        features/
          default_extractor.py
          alt_extractor.py                   ← required second implementation
        judges/
          gemini_judge.py
          mock_judge.py                      ← default for tests/local dev
        datasets/
          halueval_loader.py
          truthfulqa_loader.py
          simpleqa_loader.py
      fingerprint/
        cluster.py
        calibrate.py
      gate/
        rules.py
        graph.py                             ← PyReason integration
        score.py
      jobs/
        background.py                        ← calibration job runner
      config.py                              ← plugin selection, config paths
    tests/
  frontend/
    src/
      pages/
        ScoreResponse.tsx
        ModelFingerprints.tsx
        Evaluation.tsx
      components/
        FingerprintRadar.tsx                 ← signature radar component
        VerdictBand.tsx                      ← G on T_L–T_H scale
        FeatureBar.tsx
        Disclaimer.tsx
      lib/
        api.ts                               ← typed /api/v1/* client
        tokens.ts                            ← design tokens (single source of truth)
      styles/
        tailwind.config.ts                   ← 8 colors + 3 typefaces, NOT defaults
    tests/
  data/
    raw/
    processed/
  eval/
    metrics.py
    baselines.py
    run_experiment.py
  configs/                                    ← YAML: experiment + plugin-selection
  docs/
    decisions.md
    Hallucination_Fingerprinting_Master_Spec_v2.md
  README.md
  .gitignore
  .env.example
  pyproject.toml
```

---

## 13. Files Phase B Should NOT Touch

| Path | Reason |
|---|---|
| `/home/rishil-n/Documents/Interview Perp/Claude Text v2.md` | External spec source — read-only reference, do not modify |
| `/home/rishil-n/Documents/Interview Perp/AI Project.md` | Interview prep notes — not part of the project |
| `/home/rishil-n/hallucination-review/` | Literature review — separate project |
| `/home/rishil-n/review-paper/` | Paper project — separate project |
| `/home/rishil-n/.config/Code/Backups/` | VS Code internals — never modify |
| Any file outside `/home/rishil-n/AI_PROJECT/` | Out of workspace scope |

---

## Confirmed Design Token Reference (for Phase B)

These are the **exact, verified values** from [Section 5.13, lines 259–274](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L259-L274):

**Colors (8 total — no others permitted):**

| Token | Hex | Purpose |
|---|---|---|
| `--graphite-950` | `#0B0D10` | Base background |
| `--graphite-800` | `#14171C` | Panel/surface background |
| `--graphite-600` | `#262B33` | Borders, dividers, inactive states |
| `--paper-100` | `#E8EAED` | Primary text |
| `--paper-400` | `#8B93A1` | Secondary/muted text |
| `--signal-teal` | `#4FE3C1` | LOW RISK / clear / calibrated-normal |
| `--signal-amber` | `#F2B84B` | AMBIGUOUS / escalation-in-progress |
| `--signal-coral` | `#FF6B4A` | HIGH RISK / flagged |

**Typography (3 faces):**

| Face | Role | Usage Rule |
|---|---|---|
| Space Grotesk | Display | Hero statements and section headers only |
| Inter | Body | Prose and UI copy |
| JetBrains Mono | Data/telemetry | Exclusively for measured values (scores, thresholds, model IDs, timestamps) |

---

## Confirmed API Contract Reference (for Phase B)

Verified against [Section 5.12, lines 180–216](file:///home/rishil-n/Documents/Interview%20Perp/Claude%20Text%20v2.md#L180-L216):

```
POST /api/v1/score
  Request:  { "answer": str, "model_id": str }
  Response: {
    "verdict": "LOW_RISK" | "HIGH_RISK" | "RESOLVED_AMBIGUOUS",
    "gate_score": float,
    "thresholds": { "t_low": float, "t_high": float },
    "resolved_by": "gate" | "llm_judge",
    "triggered_patterns": [{ "name": str, "features_involved": [str], "strength": float }],
    "feature_breakdown": { "H": float, "S": float, "C": float, "E": float, "D": float, "M": float },
    "explanation": str,
    "explanation_source": "symbolic" | "llm_judge",
    "model_id": str,
    "fingerprint_version": str
  }

GET  /api/v1/models                         → list with fingerprint_version, calibration_dataset_size, last_calibrated_at
GET  /api/v1/models/{model_id}/fingerprint   → centroids, (t_low, t_high), w_i
POST /api/v1/models/{model_id}/calibrate     → { "job_id": str }
GET  /api/v1/jobs/{job_id}                   → { "status": "pending"|"running"|"complete"|"failed", "result": ... | null }
GET  /api/v1/eval/summary                   → AUROC, AUPRC, ECE, escalation_rate per model + baseline comparison
GET  /api/v1/health                          → liveness check
```

---

## Open Questions for Human Decision (before Phase B starts)

1. **Pip installation**: May I run `sudo apt-get install python3-pip` to unblock Python package installation?
2. **Python version**: Should I attempt PyReason on 3.14 first, or proactively install Python 3.10 and use a virtualenv?
3. **Dataset download**: Should I download HaluEval/TruthfulQA/SimpleQA via bypass-sandbox network access, or will you provide the files?
4. **RESOLVED_AMBIGUOUS direction**: Should the API response include a structured field for the judge's resolved direction (e.g., `"resolved_direction": "low_risk" | "high_risk"`), or is the natural-language `explanation` sufficient?

---

## PHASE A STATUS: READY FOR PHASE B
