# Hallucination Fingerprinting — Master Engineering Specification (v2, Full-Stack)

### Model-Calibrated Neuro-Symbolic Decision Gate — ML Pipeline + FastAPI Backend + Frontend

Prepared for: Rishil (TAM — The AI and ML Club, VIT Vellore) Purpose: Definitive, end-to-end project specification — research pipeline, plugin-based backend, and a genuinely distinctive frontend — plus a single copy-paste-ready master prompt for an AI coding agent (Cline).

This is v2 of the spec. It supersedes the earlier ML-only version by adding a full-stack application layer (Sections 5.12–5.14, updated Section 7 repo layout, updated Section 8 team, and a much longer Section 11 master prompt). Sections 1–4, 6, 9–10 are carried over unchanged from the validated v1 reasoning; Section 5 is extended, not replaced.

---

## SECTION 1 — PROJECT UNDERSTANDING

**Problem statement.** Existing hallucination detectors treat every model identically and rely on either (a) expensive fact-verification against a knowledge base, requiring ground truth per query, or (b) opaque internal-state methods (logits, hidden activations, latent embeddings) that are uninterpretable and often require white-box model access. This project instead treats hallucination as a **per-model behavioral phenomenon**: each LLM has a measurable, characteristic linguistic "tell" — a signature pattern in hedging, specificity, citation phrasing, and confidence markers — that correlates with when it is fabricating rather than recalling. The system detects hallucination risk from **black-box text alone**, using a three-stage pipeline that escalates cost only when necessary, exposed through a real application (API + UI), not just a notebook.

**Inputs.** A single LLM-generated answer to a factual question, plus an identifier for which model produced it (so the correct fingerprint/gate is loaded). At the application layer, inputs arrive via HTTP requests (pasted text, or a batch file upload) rather than only as in-memory Python objects.

**Outputs.** One of three verdicts — LOW RISK, HIGH RISK, or (internally) AMBIGUOUS-then-resolved — plus a structured explanation naming which behavioral signal(s) triggered the classification. At the application layer, this is returned as JSON from the API and rendered visually in the frontend (see Section 5.13).

**Prediction target.** Response-level hallucination risk in v1. Claim-level decomposition remains a documented extension, not a v1 requirement — this discipline does not change just because a frontend is being added; scope creep into claim-level UI (e.g., highlighting individual sentences) is explicitly deferred to v2 of the product, not silently absorbed into v1's timeline.

**Unit of analysis, confidence semantics, evidence representation, uncertainty representation, system boundaries** — unchanged from v1 (see below, condensed): one full model response to one factual question; `G` is a defined, calibrated quantity, not a native probability, unless a calibration step is actually run; no external knowledge base is used, "evidence" means extracted behavioral telemetry; uncertainty is represented structurally via the AMBIGUOUS band between `T_L` and `T_H`.

**New system boundary for v2 (full-stack):** the API and frontend are a **research demonstration and evaluation interface**, not a production SaaS product. This must be stated plainly in the UI itself (a visible, honestly worded disclaimer, not a legal footer buried in fine print) — the system is not adversarially robust, is not fact-checking, and is a research artifact. Aesthetic polish should not imply a production-grade guarantee the system doesn't have.

**Explicitly identified ambiguities (resolved by assumption, extended for v2):**

1. Feature set: six features — H, S, C, E, D, M (defined in Section 5.3, unchanged from v1).
2. Threshold/weight learning: grid/Bayesian search on a validation split (Section 5.6/6, unchanged).
3. Ground truth source: HaluEval primary, TruthfulQA/SimpleQA secondary (Section 6, unchanged).
4. Judge context: LLM judge sees which rules were ambiguous, not just the raw answer (unchanged).
5. **New — API authentication**: v1 of the API assumes a single-user, local/demo deployment context (no multi-tenant auth system). API keys for the underlying Gemini calls are server-side only, never exposed to the frontend. This is an explicit, documented limitation, not an oversight.
6. **New — plugin scope**: plugins are for swapping _interchangeable components_ (feature extractors, judges, dataset loaders) behind a stable interface — not a general-purpose third-party extension marketplace. Do not over-engineer the plugin system into something bigger than the project needs.

---

## SECTION 2 — CURRENT STATE OF THE ART

_(Unchanged from v1 — carried over for completeness. Items marked "verified" were confirmed via search during this project's research phase; items marked "general knowledge" are standard field background and should be spot-checked before being cited in any paper or patent filing.)_

|Family|Representative approach|Access needed|Interpretable?|Per-model calibrated?|
|---|---|---|---|---|
|Retrieval/fact verification|KB lookup, RAG-grounded claim check|Black-box|Partially|No|
|Self-consistency|SelfCheckGPT, SAC³ (sample N outputs, check agreement) — _general knowledge_|Black-box|No|No|
|Contradiction/NLI-based|ConFactCheck-style fact-alignment + uniform-distribution check — _verified_|Black-box + token probs|Partially|No|
|Internal-representation|Eigenvalue analysis of hidden activations (LLM-Check) — _verified_; entropy-distribution fingerprinting — _verified_|White-box|No|Yes, but opaque|
|Latent-embedding|Vodafone VAE patent: output vector → latent space anomaly, no ground truth needed — _verified_|Black-box|No|Unclear from public summary|
|Cascade/regeneration|Google patent: detect hallucination, regenerate, verify second response — _verified_|Black-box|Partially|Unclear|
|Lexical/stylistic (general)|TRACT: hedging trends + step-length dynamics for long-form reasoning — _verified_|Black-box|Yes|No (global scorer)|
|Model fingerprinting (different purpose)|Llmmap, CoTSRF, DuFFin, SRAF, Copyleaks — model _identification_, not hallucination — _verified_|Black-box|Partially|Yes, wrong target variable|
|Governance/composite scoring|LTI MindTree, Accenture, ServiceNow, BMC, Oracle filings — dashboards — _verified_|N/A|N/A|N/A|

**The gap this project targets:** an interpretable, feature-based, **per-model-adaptive** gate sitting before an LLM judge, where adaptivity applies to the escalation decision boundary itself, not just cluster centroids. Nothing found combines all three properties (interpretable + per-model-adaptive + targets hallucination specifically, not authorship).

---

## SECTION 3 — NOVELTY OPTIONS

_(Unchanged from v1; carried over for completeness.)_

**Option A — Per-model fingerprint + LLM judge (2-stage).** Necessary infrastructure, not the differentiator on its own. Patentability weak alone.

**Option B — Neuro-symbolic gate with a fixed global threshold.** Rejected as final form — a single global threshold can't account for models with structurally different baseline hedging/specificity rates.

**Option C — Model-Calibrated Neuro-Symbolic Decision Gate (adaptive per-model thresholds `T_L`, `T_H`, and rule weights `w_i`) — SELECTED.** Directly fixes Option B's failure mode. Central falsifiable hypothesis: adaptive per-model thresholds achieve a better AUROC-vs-escalation-rate Pareto frontier than a global threshold fit across all models pooled. Full detail in Section 4.

**Rejected without full write-up:** reinforcement learning for threshold tuning (no evidence it would outperform simpler calibration here); full claim-level decomposition (separate, larger project — documented future work, not v1 scope).

---

## SECTION 4 — SELECTED CORE CONTRIBUTION

**The Model-Calibrated Neuro-Symbolic Decision Gate.** A three-stage architecture: (1) black-box behavioral telemetry extraction, (2) a deterministic symbolic rule graph evaluating that telemetry against per-model-calibrated thresholds to produce a bounded anomaly score `G`, cheaply resolving clear-cut cases with no LLM call, and (3) a secondary LLM judge invoked only when `G` falls in the model-specific AMBIGUOUS band, receiving the specific ambiguous rule activations as targeted context.

**Central claim:** making the escalation boundary itself part of the per-model fingerprint — not a fixed global threshold — improves the cost/accuracy tradeoff, because different models have measurably different baseline behavioral distributions.

This remains the single coherent idea the entire system (research pipeline _and_ application layer) exists to test and demonstrate. The frontend's job is to make this specific mechanism visible and legible to a viewer — not to become a generic "AI dashboard" that could belong to any project.

---

## SECTION 5 — FINAL SYSTEM ARCHITECTURE

### 5.1 Data ingestion

- **Responsibility**: load QA pairs + model responses + ground-truth hallucination labels.
- **Input**: HaluEval is the **primary** source (`github.com/RUCAIBox/HaluEval`), specifically the **QA subset** (`qa_data.json` — correct response, hallucinated response, supporting knowledge), not the dialogue subset, since the project targets single-turn factual QA, not multi-turn conversation. TruthfulQA (`huggingface.co/datasets/truthfulqa/truthful_qa`) and SimpleQA are **secondary/evaluation-only**.
- **Output**: normalized `Record{question, answer, model_id, label, source_dataset, dataset_role}`, where `dataset_role ∈ {"primary","secondary"}` is required on every record.
- **Failure modes**: label schema mismatch; accidental pooling of secondary data into calibration splits (mitigated by a required, tested filter — see Section 7).

### 5.2 Dataset normalization

- HaluEval QA subset labels map directly (already a hallucinated/correct response pair per question). TruthfulQA's correct/incorrect reference answers are unrolled into a binary label and explicitly commented in code as a _factuality_ proxy, not a clean hallucination label (see Section 6 rationale — TruthfulQA often reflects human-misconception errors rather than fabrication). SimpleQA gets its own explicit mapping documented at implementation time. Every mapping is dataset-specific and code-reviewed individually.

### 5.3 Feature extraction (the telemetry layer)

Six features, computed deterministically from response text — no LLM call:

|Feature|Definition (v1)|Method|
|---|---|---|
|H — hedge density|hedge-phrase count / 100 words|lexicon match|
|S — specificity|ratio of named entities + numbers/dates to total tokens|spaCy NER + regex|
|C — citation vagueness|source-gesturing phrases not followed by a named source within N tokens|pattern match + proximity check|
|E — evidence density|concrete, checkable factual assertions per sentence|entity + relation co-occurrence heuristic|
|D — semantic/entity drift|embedding-similarity variance across sentences|sentence embeddings + pairwise cosine variance|
|M — confidence-marker density|absolute/confidence words / 100 words|lexicon match|

Each feature is normalized to `[0,1]` per model using that model's own training-set distribution. This per-model normalization is itself part of what makes the fingerprint per-model, distinct from the downstream threshold adaptivity.

### 5.4 Fingerprint construction (offline, per model)

- k-means or GMM clustering of labeled feature vectors, separately per model, into hallucination/correct style regions. Output: a versioned, serializable fingerprint file per model (centroids, normalization parameters, later augmented with calibrated weights/thresholds in 5.6).

### 5.5 Symbolic rule graph (PyReason)

Facts derived from normalized features via calibrated interval annotation:

```
hedging_high(answer)    : [0.82, 1.00]
specificity_high(answer): [0.76, 1.00]
citation_vague(answer)  : [0.91, 1.00]
evidence_low(answer)    : [0.70, 1.00]
```

Relational pattern rules:

```
anomaly_pattern_1(X) <- hedging_high(X), citation_vague(X)
anomaly_pattern_2(X) <- specificity_high(X), evidence_low(X)
anomaly_pattern_3(X) <- confidence_high(X), entity_drift_high(X)
```

Aggregate score: `G = Σ w_i * r_i`, where `r_i` is rule activation strength and `w_i` is a learned, per-model weight. **Preserved correction**: PyReason provides annotated real-valued bounds on facts/rule conclusions, not an automatically calibrated hallucination probability — `G` is defined and calibrated by this system, not supplied natively. This must appear in any paper/patent draft and in the frontend's own explanatory copy (see 5.13) — do not let UI copy imply `G` is a native, out-of-the-box probability.

### 5.6 Threshold and weight calibration (offline, per model)

- `w_i` and per-model `(T_L, T_H)` learned on a validation split by optimizing the joint objective in Section 6 — never chosen by inspection.
- Verdict logic:

```
G < T_L         → LOW RISK      (accept, no escalation)
T_L ≤ G ≤ T_H   → AMBIGUOUS     (escalate to LLM judge)
G > T_H         → HIGH RISK     (flag, no escalation)
```

### 5.7 LLM judge escalation (Stage 3, Gemini)

- Invoked only on AMBIGUOUS verdicts. Prompt includes the answer text, the model's fingerprint summary, and specifically which rule(s) produced the ambiguous activation.
- **Output**: final risk verdict + natural-language explanation, returned as structured JSON (not free text) so the frontend can render it reliably — see 5.12 for the exact schema.

### 5.8 Explainability layer

- Every verdict returns: `{verdict, G, triggered_patterns, resolved_by: "gate"|"judge", explanation, feature_breakdown: {H,S,C,E,D,M}}`. Never presents unverifiable LLM chain-of-thought as ground truth — the judge's explanation is clearly labeled as model-generated reasoning, not fact, both in the API schema (`explanation_source: "symbolic"|"llm_judge"`) and in the UI copy.

### 5.9 Logging & experiment tracking

- All runs versioned (config, fingerprint version, code commit hash). Lightweight tooling (structured JSON logs or MLflow) — no custom infrastructure beyond what's needed.

### 5.10 Evaluation pipeline

See Section 6 in full.

### 5.11 Deployment assumptions (revised for v2)

The application (API + frontend) is deployable as a local or lightly-hosted demo (e.g., `uvicorn` locally, or a single free-tier deployment such as Render/Railway/Fly.io for a live demo link during a presentation) — it is explicitly **not** designed, claimed, or engineered as a scalable, multi-tenant production system. No horizontal scaling, no queueing infrastructure, no production auth system are in scope. State this plainly in both the README and the in-app disclaimer.

---

### 5.12 API Layer (FastAPI)

**Responsibility.** Expose the three-stage pipeline over HTTP, in a way a frontend (or a curl command, or a grader) can call directly, with a plugin registry so components can be swapped without touching call sites.

**Design principles for this layer:**

- The API is the single source of truth for verdicts — the frontend never runs pipeline logic client-side, it only calls the API and renders what comes back. This keeps the research logic in one place, testable independently of any UI concern.
- Every response follows one strict Pydantic schema (below) — no ad hoc JSON shapes per endpoint.
- Long-running operations (fingerprint construction/calibration for a new model) are separated from fast operations (scoring a single answer) — the former is a background job with a status-poll endpoint, the latter is a synchronous request/response.

**Core endpoints:**

```
POST /api/v1/score
  Request:  { "answer": str, "model_id": str }
  Response: {
    "verdict": "LOW_RISK" | "HIGH_RISK" | "RESOLVED_AMBIGUOUS",
    "gate_score": float,                # G
    "thresholds": { "t_low": float, "t_high": float },   # this model's calibrated band
    "resolved_by": "gate" | "llm_judge",
    "triggered_patterns": [ { "name": str, "features_involved": [str], "strength": float } ],
    "feature_breakdown": { "H": float, "S": float, "C": float, "E": float, "D": float, "M": float },
    "explanation": str,
    "explanation_source": "symbolic" | "llm_judge",
    "model_id": str,
    "fingerprint_version": str
  }

GET  /api/v1/models
  Response: list of model_ids that have a calibrated fingerprint available, each with basic
  metadata (fingerprint_version, calibration_dataset_size, last_calibrated_at).

GET  /api/v1/models/{model_id}/fingerprint
  Response: the model's fingerprint summary — cluster centroid positions per feature, calibrated
  (T_low, T_high), rule weights (w_i) — this is what powers the frontend's fingerprint visualization
  (see 5.13's signature element). Never returns raw training data, only the derived fingerprint.

POST /api/v1/models/{model_id}/calibrate
  Triggers a background calibration job (Stage 1 clustering + Stage 2 threshold/weight fitting)
  against the primary dataset. Returns a job_id immediately.

GET  /api/v1/jobs/{job_id}
  Response: { "status": "pending"|"running"|"complete"|"failed", "result": ... | null }

GET  /api/v1/eval/summary
  Response: the latest stored evaluation metrics (AUROC, AUPRC, ECE, escalation_rate per model,
  plus the baseline comparison table from Section 6) — powers the frontend's evaluation view.

GET  /api/v1/health
  Basic liveness check.
```

**Plugin architecture (this satisfies the explicit "plugins" requirement — keep it this scoped, not bigger):**

Three plugin interfaces, each a simple Python ABC or Protocol, registered in a small in-process registry (a dict keyed by plugin name — do not reach for a heavyweight plugin-discovery framework for a project this size):

```python
class FeatureExtractorPlugin(Protocol):
    name: str
    def extract(self, answer: str) -> dict[str, float]:  # returns {"H":..., "S":..., ...}
        ...

class JudgePlugin(Protocol):
    name: str
    def judge(self, answer: str, fingerprint_summary: dict, ambiguous_patterns: list[dict]) -> JudgeResult:
        ...

class DatasetLoaderPlugin(Protocol):
    name: str
    def load(self) -> Iterable[Record]:
        ...
```

- The default `FeatureExtractorPlugin` implements the six features in 5.3. A second, optional plugin (e.g., an alternate lexicon or a different NER backend) demonstrates the interface is real, not decorative — build at least one alternate implementation, even a simple one, so "plugin" isn't just an unused abstraction.
- The default `JudgePlugin` calls Gemini. A `MockJudgePlugin` (returns a deterministic canned response) must also exist and be the default in tests and local dev without an API key — this avoids tests silently depending on a live API key being present, which is a common source of flaky CI.
- `DatasetLoaderPlugin` implementations exist for HaluEval, TruthfulQA, SimpleQA, each tagging `dataset_role` correctly per Section 5.1's rule.
- Plugins are selected via configuration (a YAML/env value naming which plugin to use per component), not hardcoded imports scattered through the codebase — this is what makes it a genuine plugin system rather than a rebranded if/else chain.

**What NOT to build**: user accounts, billing, rate limiting beyond a basic in-memory guard, a general third-party plugin marketplace, WebSocket streaming (polling is sufficient at this scale). Building any of these without being asked is scope creep and violates the project's explicit anti-complexity principle.

---

### 5.13 Frontend Layer — Aesthetic Direction

**Framing.** This is not a generic admin dashboard. The subject is genuinely visual and specific: a model's "fingerprint" is, literally, a signature pattern — treat that literally rather than defaulting to a generic SaaS-analytics look (no cream background + terracotta accent, no near-black + acid-green, no broadsheet hairline-rule template — these are the current AI-generated-design defaults and this brief has enough of a real, specific subject to earn something distinctive instead).

**Design concept: "Signal Forensics."** The system reads a model's writing the way a forensic analyst reads a signature or a voiceprint — looking for the involuntary tells underneath the surface content. The UI should feel like a forensic instrument: precise, technical, slightly clinical, with data treated as physical evidence rather than decoration.

**Token system:**

_Color_ (named, not generic):

- `--graphite-950: #0B0D10` — base background, a true near-black ink, not pure #000
- `--graphite-800: #14171C` — panel/surface background
- `--graphite-600: #262B33` — borders, dividers, inactive states
- `--paper-100: #E8EAED` — primary text (deliberately named "paper" against the "ink" background — the forensic-document metaphor)
- `--paper-400: #8B93A1` — secondary/muted text
- `--signal-teal: #4FE3C1` — primary accent: LOW RISK / clear signal / calibrated-confidence state
- `--signal-amber: #F2B84B` — AMBIGUOUS state / escalation-in-progress
- `--signal-coral: #FF6B4A` — HIGH RISK / flagged state

Do not introduce additional accent hues beyond these three semantic signal colors — the discipline of "three meanings, three colors, nothing else competing for attention" is the point, and mirrors the actual three-way verdict logic of the system itself (structure encoding real information, not decoration).

_Typography:_

- Display face: **Space Grotesk** (geometric, slightly technical, has real personality without being a default choice) — used with restraint, primarily for the hero statement and section headers.
- Body face: **Inter** — quiet, legible, does not compete with the display face.
- Data/telemetry face: **JetBrains Mono** — used specifically for anything that is a _measurement_: feature values, gate scores, thresholds, model IDs, timestamps. This typographic distinction (mono for data, humanist sans for prose) is itself a piece of information design: it tells the viewer "this number is measured evidence" vs. "this is explanation," reinforcing the forensic framing without needing a label.

_Layout concept:_ a **scan report**, not a dashboard grid. The primary score view for a single answer reads top to bottom like a lab result: input → extracted telemetry (six features, shown as a horizontal bar per feature with the model's calibrated normal range marked) → the gate's verdict with its score plotted on a line between `T_L` and `T_H` for that specific model → (if escalated) the judge's explanation, visually subordinate to and clearly separated from the gate's own output, never blended together as if from the same source.

**Signature element (the one memorable thing, per design methodology — spend the boldness here, keep everything else quiet): the Fingerprint Radar.**

A radial/polar plot with six axes (H, S, C, E, D, M), rendered to visually resemble a fingerprint's ridge pattern rather than a generic radar chart:

- The model's _calibrated normal region_ (bounded by that model's per-feature ranges from its fingerprint) is drawn as a soft, filled, slightly irregular ring — not a perfect polygon — using layered, slightly offset concentric paths at low opacity to genuinely evoke ridge lines, colored `--signal-teal` at low opacity.
- The _current answer's_ feature vector is plotted as a single sharp line on top of that ring, colored by its verdict state (teal/amber/coral).
- Where the current line pushes outside the calibrated ring, that stretch of the line is visually emphasized (slightly thicker, brighter) — this is the "which tell fired" explanation made visual, not just textual, directly reinforcing the explainability principle from Section 5.8 rather than being decorative.
- This same visualization, shown per-model with no current-answer overlay, is what powers the `/models/{model_id}/fingerprint` view — i.e., it's a genuine reusable component representing the actual data structure, not a one-off hero graphic.

**Motion:** restrained. A single deliberate moment — when a verdict resolves, the Fingerprint Radar's current-answer line draws itself stroke-by-stroke (a signature literally being "signed" onto the page) over roughly 600–800ms, rather than snapping in instantly. No other animation beyond this and ordinary, non-showy hover/focus states. Respect `prefers-reduced-motion` — fall back to an instant, non-animated render.

**Pages/views required:**

1. **Score a Response** (primary/home view) — paste an answer, pick a model, submit, see the scan-report layout described above, ending in the Fingerprint Radar with the current answer overlaid.
2. **Model Fingerprints** — browse calibrated models, view each one's baseline Fingerprint Radar and calibration metadata (dataset size, last calibrated, `T_L`/`T_H`).
3. **Evaluation** — the honest research view: AUROC/AUPRC/ECE per model, escalation rate, and the adaptive-vs-global-threshold comparison chart that is the actual research claim of this project — this page exists specifically to make Section 6's core experiment legible to a viewer, not as a generic "analytics" page.
4. A visible, plainly worded **research-demo disclaimer** (persistent, not a dismissible modal) — reflecting the honesty principle from Section 5.11 and Section 10's conservative patent stance: this is a research prototype, not a production fact-checking guarantee.

**Copy/voice:** plain, technical, no marketing language. Verdicts are stated as what they are ("Gate score 0.74 — within this model's ambiguous band (0.61–0.79), escalated for review"), not dramatized ("Uh oh, looks fishy!"). Empty/error states explain what happened and what to do next, in the interface's own voice.

**Stack recommendation:** React + Tailwind for the frontend (Tailwind config extended with the exact token values above, not default Tailwind palette), calling the FastAPI backend via `fetch`. Keep the frontend a single cohesive app, not a component-library showcase — build only the components this spec actually describes.

---

### 5.14 Plugin Architecture — Summary Cross-Reference

Covered fully in 5.12. Restated here only to note the design/engineering split: the _frontend_ never needs to know a plugin system exists — it only ever talks to the stable `/api/v1/*` contract in 5.12. Plugin swaps happen entirely server-side via configuration. This separation must be preserved; do not let frontend code branch on which plugin is active.

---

## SECTION 6 — RESEARCH PLAN

_(Unchanged from v1 — carried over for completeness, since the application layer consumes these results via `/api/v1/eval/summary` and must not duplicate or reinterpret them.)_

**Baselines** (all must be implemented, not just cited):

1. Always-escalate (every answer to the LLM judge).
2. Never-escalate / gate-only.
3. Global-threshold gate (single `(T_L, T_H)` pooled across models — the direct ablation target).
4. Random/majority-class classifier.
5. Generic (non-per-model) lexical classifier, reproducing TRACT-style pooled hedging features.

**Datasets:**

|Dataset|Role|Size|Source|
|---|---|---|---|
|**HaluEval (QA subset)**|**Primary**|~10,000 samples (correct response, hallucinated response, supporting knowledge)|`github.com/RUCAIBox/HaluEval` → `data/qa_data.json`|
|**TruthfulQA**|Secondary/adversarial|817 questions, 38 categories|`huggingface.co/datasets/truthfulqa/truthful_qa`|
|**SimpleQA**|Secondary|short-answer QA pairs|Hugging Face / OpenAI release|
|**MedHallu** (optional)|Out-of-domain robustness only|1,000 human-labeled + 9,000 synthetic|`huggingface.co/datasets/UTAustin-AIHealth/MedHallu`|

Rationale for HaluEval-primary: a 2026 benchmark analysis (HalluLens) argues TruthfulQA reflects human misconception/factuality errors more than fabrication-style hallucination, and includes time-sensitive prompts whose correct answer changes — a different failure mode from what H/S/C/E/D/M are designed to detect. HaluEval's labels are constructed specifically around hallucination, matching this project's target variable directly.

**Metrics:** AUROC, AUPRC (primary given expected class imbalance), ECE (required if `G` is ever presented as a probability anywhere, including in the frontend), escalation rate, latency per verdict, cost per verdict (LLM calls avoided).

**Core experiment:** per-model adaptive `(T_L, T_H, w_i)` vs. global pooled `(T_L, T_H, w_i)`, same features, same rule structure, reported as a Pareto frontier of AUROC vs. escalation rate across ≥2–3 distinct models.

**Ablations:** feature-subset ablation; learned vs. uniform `w_i`; cluster-count sensitivity.

**Sensitivity analysis:** how much labeled data per model is required before adaptive thresholds outperform global ones.

**Reproducibility:** fixed seeds, versioned splits, documented hyperparameter search ranges, code+config committed alongside every reported number.

---

## SECTION 7 — ENGINEERING PLAN

**Repository structure (v2, full-stack):**

```
hallucination-gate/
  backend/
    app/
      main.py                 # FastAPI app entrypoint
      api/
        routes_score.py
        routes_models.py
        routes_jobs.py
        routes_eval.py
      schemas.py               # Pydantic response/request models (Section 5.12 contract)
      plugins/
        registry.py
        features/
          default_extractor.py
          alt_extractor.py     # second implementation, proves the interface is real
        judges/
          gemini_judge.py
          mock_judge.py        # default for tests/local dev
        datasets/
          halueval_loader.py
          truthfulqa_loader.py
          simpleqa_loader.py
      fingerprint/
        cluster.py
        calibrate.py
      gate/
        rules.py
        graph.py
        score.py
      jobs/
        background.py          # calibration job runner + status store
      config.py                 # plugin selection, thresholds config path
    tests/
  frontend/
    src/
      pages/
        ScoreResponse.tsx
        ModelFingerprints.tsx
        Evaluation.tsx
      components/
        FingerprintRadar.tsx    # the signature element, reused across pages
        VerdictBand.tsx         # G plotted between T_L/T_H
        FeatureBar.tsx
        Disclaimer.tsx
      lib/
        api.ts                  # typed client for the /api/v1/* contract
        tokens.ts                # design tokens from Section 5.13, single source of truth
      styles/
        tailwind.config.ts       # extended with Section 5.13's exact palette/type tokens
    tests/
  data/
    raw/
    processed/
  eval/
    metrics.py
    baselines.py
    run_experiment.py
  configs/                       # versioned experiment + plugin-selection configs (yaml)
  docs/
    decisions.md
    Hallucination_Fingerprinting_Master_Spec_v2.md   # this file
  README.md
```

**Implementation sequence (strict order):**

1. Data ingestion + normalization, HaluEval QA subset first (tests validating label mapping and `dataset_role` filtering before TruthfulQA/SimpleQA are added).
2. Feature extraction (unit tests on hand-crafted example sentences with known expected feature values).
3. Baselines 1 and 4 — get the evaluation harness working end-to-end before any novel component exists.
4. Fingerprint clustering (Stage 1).
5. Symbolic gate with a global fixed threshold first (Baseline 3) — the ablation baseline before the full contribution.
6. Per-model adaptive calibration (the core research contribution).
7. LLM judge integration + escalation logic (`MockJudgePlugin` first, `GeminiJudgePlugin` second).
8. Full pipeline evaluation + ablations, results written to a location `/api/v1/eval/summary` can read.
9. **Backend API** — wrap the now-working pipeline in FastAPI routes per Section 5.12. Do not build the API before the pipeline it wraps is real; an API around a stubbed pipeline invites drift between what the UI demos and what the research actually shows.
10. **Frontend** — build against the real API, in the order: Score a Response → Model Fingerprints → Evaluation → Disclaimer/shell polish. Use Section 5.13's tokens from the start; do not build with default Tailwind colors and "re-skin later" — the aesthetic direction should inform component structure from the first component, not be applied as a coat of paint at the end.
11. Documentation and write-up.

---

## SECTION 8 — MULTI-AGENT TEAM DESIGN

|Role|Owns|Produces|Reviews|Must never decide alone|
|---|---|---|---|---|
|Principal Architect|overall system design, scope discipline|architecture docs, ADRs|all major structural changes|scope expansion beyond Section 4's core claim|
|Research Lead|experiment design, metric choice|research plan, ablation design|statistical claims before write-up|whether a result counts as "novel"|
|ML Engineer|feature extraction, gate, calibration code|working modules + tests|Research Lead's experiment design for implementability|final threshold values (must come from calibration)|
|Backend Engineer|FastAPI routes, plugin registry, job system|API + plugin code + tests|API contract changes against Section 5.12|changing the response schema without updating the frontend contract in the same change|
|Frontend/UX Engineer|component implementation of Section 5.13's direction|React components, Tailwind tokens|any visual decision not derivable from the token system|introducing a new accent color or typeface outside the token system|
|Evaluation Scientist|metrics correctness, leakage checks|evaluation reports|every reported number before use in a claim or in the Evaluation page|whether an ablation "proves" the hypothesis|
|Red-Team/Adversarial Engineer|stress-testing the gate|adversarial test cases|robustness claims, including any UI copy that might overstate robustness|—|
|Code Reviewer|code quality, maintainability|review comments|every PR/commit|merging own unreviewed work|

**Interaction principle unchanged**: no role self-certifies its own output. The Frontend/UX Engineer in particular does not get to unilaterally decide the aesthetic direction diverges from Section 5.13 — that's an Architect-level decision, logged in `docs/decisions.md` like any other.

---

## SECTION 9 — FAILURE / SECURITY / ADVERSARIAL STRATEGY

_(Core taxonomy unchanged from v1; extended with application-layer failure modes.)_

**Pipeline-level** (unchanged): cold-start (no fingerprint for a queried model — must explicitly refuse or fall back with a flagged low-confidence warning, never silently apply another model's thresholds); fingerprint drift; lexicon brittleness to paraphrase; threshold overfitting on small validation sets; correlated features reducing the graph's real independent evidence (check via correlation analysis before claiming the multi-pattern graph adds value over a single composite score).

**Application-layer (new for v2):**

- _Backend_: the Gemini API key must never be sent to or accessible from the frontend — server-side only, loaded from environment/secrets, never logged. The `/calibrate` endpoint (a potentially expensive operation) needs at minimum a basic guard against being triggered repeatedly/accidentally (e.g., reject a new calibration job for a model while one is already running for it).
- _Frontend_: never fabricate a plausible-looking result while a request is loading — use honest loading states, not skeleton screens that could be mistaken for a real (but empty) result.
- _Adversarial gaming_: this is a style-only detector; a crafted answer (artificially low hedging + high fake specificity) could slip under `T_L`. This limitation must be stated in the Disclaimer view, not just in engineering docs — the honesty principle applies to the UI's own copy, not only to internal documentation.

---

## SECTION 10 — PATENT / NOVELTY ASSESSMENT

_(Unchanged, conservative, carried over from v1.)_

The adaptive-threshold-as-fingerprint mechanism remains the strongest single claim element identified, and does not appear in any patent or paper found to date. However: no professional prior-art search has been conducted (only web search and summarized patent-landscape posts — a hard blocker on any confident patentability claim); zero reduction to practice existed as of the prior version of this document — **this v2 spec, by requiring the core experiment (Section 6) to run before the API is built (Section 7, step 9), structurally forces reduction to practice to happen before the demo layer exists**, which is a meaningful improvement in patent posture, not just an engineering nicety. Non-obviousness risk remains real: every individual component (symbolic gating, per-model calibration, cascading verification) exists somewhere in the literature; the claim rests on the specific combination plus the adaptive-threshold twist.

Recommended path unchanged: file a provisional now to lock priority date on the architecture; treat the core experiment as urgent; commission a real prior-art search before the complete specification is due; narrow claims into one strong independent claim (adaptive threshold mechanism) plus dependent claims (feature set, graph structure, escalation logic).

The frontend/backend application is **evidence of reduction to practice and a demonstration vehicle** — it is not itself the subject of the patent claim. Do not conflate "we built a nice UI" with "we have a stronger patent" in any pitch; the UI's value is making the underlying mechanism legible and testable to a human reviewer, nothing more, nothing less.

---

## SECTION 11 — FINAL MASTER PROMPT

_(Copy everything below this line into Cline or an equivalent coding agent. This supersedes the v1 master prompt — it is longer and covers the full stack deliberately; do not summarize or trim it before pasting.)_

```
You are not merely a code generator. You are the implementation arm of a senior AI research and engineering
team building a research-grade hallucination detection system with a real API and a genuinely distinctive
frontend — not a toy demo and not a generic dashboard template.

═══════════════════════════════════════════════════════════
OBJECTIVE
═══════════════════════════════════════════════════════════
Implement, end to end, the "Model-Calibrated Neuro-Symbolic Decision Gate":

RESEARCH PIPELINE
  Stage 1 (offline, per model): build a per-model behavioral fingerprint from labeled QA data via
  feature extraction + clustering.
  Stage 2 (the core contribution): a deterministic symbolic rule graph (PyReason) evaluates six
  behavioral features against PER-MODEL-CALIBRATED thresholds (T_L, T_H) and rule weights (w_i),
  producing a bounded anomaly score G. Cases with G < T_L or G > T_H resolve immediately with NO LLM
  call. Only cases where T_L ≤ G ≤ T_H are ambiguous.
  Stage 3 (escalation): ambiguous cases only go to an LLM judge (Gemini), which receives the answer,
  the model's fingerprint summary, and specifically which rule(s) triggered ambiguity, and returns a
  structured verdict + explanation.

APPLICATION LAYER
  A FastAPI backend exposing this pipeline over a stable JSON contract, with a plugin architecture
  for feature extractors, LLM judges, and dataset loaders.
  A React + Tailwind frontend implementing the "Signal Forensics" design direction specified below —
  built to be genuinely distinctive, not a default admin-dashboard look.

The central, falsifiable research claim everything serves: adaptive per-model (T_L, T_H, w_i)
outperforms a single global threshold on the AUROC-vs-escalation-rate tradeoff. Do not add components,
backend routes, or frontend pages that do not serve demonstrating or testing this claim.

═══════════════════════════════════════════════════════════
HARD CONSTRAINTS — RESEARCH
═══════════════════════════════════════════════════════════
- Never fabricate experimental results, numbers, or citations. If you have not run something, do not
  report a result for it, in code comments, docs, or UI copy.
- Never claim PyReason natively outputs a "hallucination probability." It provides annotated
  real-valued bounds on facts/rule conclusions. G is a quantity THIS system defines and calibrates —
  always describe it that way, including in API docs and frontend copy.
- Never describe G as a calibrated probability anywhere (API response naming, frontend labels) unless
  a calibration step (Platt scaling / isotonic regression on a held-out set) has actually been run.
- Do not add reinforcement learning, additional ensembles, retrieval, or other advanced methods beyond
  what is specified here without a measurable justification tied to the core claim. Prefer the
  simplest approach that could work.
- Do not implement claim-level decomposition, multi-turn handling, user accounts, billing, or
  horizontal-scaling infrastructure in v1 — explicitly out of scope. Flag and ask before expanding
  scope; do not silently expand it.
- English-only lexicon-based features are an accepted v1 limitation. Document it.
- HaluEval QA subset is the PRIMARY calibration dataset (github.com/RUCAIBox/HaluEval, qa_data.json).
  TruthfulQA and SimpleQA are SECONDARY / evaluation-only and must never be silently pooled into the
  primary calibration split — TruthfulQA's labels reflect factuality/misconception errors, not
  fabrication-style hallucination. Every Record must carry a dataset_role field ("primary" or
  "secondary"), and calibration code must filter on it explicitly. Add a test asserting this.

═══════════════════════════════════════════════════════════
HARD CONSTRAINTS — APPLICATION LAYER
═══════════════════════════════════════════════════════════
- Build the research pipeline (through Section 6/7 step 8: full pipeline evaluation) BEFORE building
  the FastAPI routes that wrap it, and build the FastAPI routes before the frontend. Do not build a UI
  against a stubbed or fake pipeline "to be wired up later" — this invites drift between what the demo
  shows and what the research actually demonstrates, and is explicitly forbidden.
- The API response schema is the contract in this prompt's "API CONTRACT" section below. Do not
  invent a different shape. If a field is genuinely needed that isn't listed, propose the addition and
  ask rather than silently extending the schema.
- The Gemini API key is server-side only. Never expose it to, log it in, or send it toward the
  frontend in any response payload.
- Plugins (feature extractors, judges, dataset loaders) are selected via configuration, not hardcoded
  imports scattered through the codebase. Build at least one alternate implementation of the feature
  extractor plugin (not just the default) so the plugin interface is proven to be real, not decorative.
  A MockJudgePlugin must exist and be the default for tests/local dev without requiring a live API key.
- The frontend NEVER runs pipeline logic client-side and NEVER branches on which backend plugin is
  active — it only calls the stable API contract and renders what comes back.
- The frontend must include a persistent, honestly worded research-demo disclaimer (not a dismissible
  modal) stating this is a research prototype, not a production fact-checking guarantee, and that the
  detector is style-based and not adversarially robust.

═══════════════════════════════════════════════════════════
API CONTRACT (FastAPI) — build exactly this
═══════════════════════════════════════════════════════════
POST /api/v1/score
  Request:  { "answer": str, "model_id": str }
  Response: {
    "verdict": "LOW_RISK" | "HIGH_RISK" | "RESOLVED_AMBIGUOUS",
    "gate_score": float,
    "thresholds": { "t_low": float, "t_high": float },
    "resolved_by": "gate" | "llm_judge",
    "triggered_patterns": [ { "name": str, "features_involved": [str], "strength": float } ],
    "feature_breakdown": { "H": float, "S": float, "C": float, "E": float, "D": float, "M": float },
    "explanation": str,
    "explanation_source": "symbolic" | "llm_judge",
    "model_id": str,
    "fingerprint_version": str
  }

GET  /api/v1/models
  -> list of model_ids with a calibrated fingerprint, each with fingerprint_version,
     calibration_dataset_size, last_calibrated_at.

GET  /api/v1/models/{model_id}/fingerprint
  -> cluster centroid positions per feature, calibrated (t_low, t_high), rule weights (w_i).
     Never returns raw training data.

POST /api/v1/models/{model_id}/calibrate
  -> triggers a background job (Stage 1 clustering + Stage 2 threshold/weight fitting against the
     PRIMARY dataset only). Returns { "job_id": str } immediately. Reject if a job for this model_id
     is already running.

GET  /api/v1/jobs/{job_id}
  -> { "status": "pending"|"running"|"complete"|"failed", "result": object | null }

GET  /api/v1/eval/summary
  -> latest stored evaluation metrics: AUROC, AUPRC, ECE, escalation_rate per model, plus the
     adaptive-vs-global-threshold baseline comparison table from the research plan.

GET  /api/v1/health
  -> basic liveness check.

═══════════════════════════════════════════════════════════
PLUGIN INTERFACES — build exactly these
═══════════════════════════════════════════════════════════
class FeatureExtractorPlugin(Protocol):
    name: str
    def extract(self, answer: str) -> dict[str, float]: ...   # {"H":.., "S":.., "C":.., "E":.., "D":.., "M":..}

class JudgePlugin(Protocol):
    name: str
    def judge(self, answer: str, fingerprint_summary: dict, ambiguous_patterns: list[dict]) -> JudgeResult: ...

class DatasetLoaderPlugin(Protocol):
    name: str
    def load(self) -> Iterable[Record]: ...   # Record includes dataset_role

Registry: a small in-process dict keyed by plugin name, configured via a YAML/env value naming which
plugin to use per component. Do not build a heavyweight plugin-discovery framework — this project's
scale does not need one.

═══════════════════════════════════════════════════════════
FRONTEND DESIGN DIRECTION — "Signal Forensics"
═══════════════════════════════════════════════════════════
Framing: the system reads a model's writing the way a forensic analyst reads a signature — looking for
involuntary tells under the surface content. The UI should feel like a precise instrument, not a
generic SaaS dashboard. Explicitly avoid these current AI-generated-design defaults: a cream
background with a terracotta accent; a near-black background with a single acid-green/vermilion
accent used generically; a broadsheet layout with hairline rules and zero border-radius used
generically. This brief has a real, specific subject — use it.

DESIGN TOKENS (use exactly these, do not substitute default Tailwind colors):
  Color:
    --graphite-950: #0B0D10   (base background)
    --graphite-800: #14171C   (panel/surface)
    --graphite-600: #262B33   (borders, dividers, inactive states)
    --paper-100:    #E8EAED   (primary text)
    --paper-400:    #8B93A1   (secondary/muted text)
    --signal-teal:  #4FE3C1   (LOW RISK / clear / calibrated-normal state)
    --signal-amber: #F2B84B   (AMBIGUOUS / escalation-in-progress state)
    --signal-coral: #FF6B4A   (HIGH RISK / flagged state)
  Do not add accent hues beyond these three semantic signal colors — three meanings, three colors,
  nothing else competing for attention. This mirrors the system's actual three-way verdict logic.

  Typography:
    Display face: Space Grotesk — used with restraint, hero statement and section headers only.
    Body face: Inter — for prose and UI copy.
    Data/telemetry face: JetBrains Mono — used specifically and only for measured values: feature
    scores, gate scores, thresholds, model IDs, timestamps. This distinction (mono = measured
    evidence, sans = explanation) is deliberate information design — preserve it consistently, do not
    use the mono face decoratively elsewhere.

  Layout concept: a "scan report," not a dashboard grid. The primary score view reads top to bottom
  like a lab result: input -> extracted telemetry (six features as horizontal bars, each showing the
  model's calibrated normal range) -> the gate's verdict with G plotted on a line between T_L and T_H
  for that specific model -> (if escalated) the judge's explanation, visually and clearly subordinate
  to and separated from the gate's own output — never blended together as if from the same source.

  SIGNATURE ELEMENT — build this as a genuine, reused component, not a one-off decoration:
  "Fingerprint Radar" — a radial/polar plot with six axes (H, S, C, E, D, M), styled to evoke a
  fingerprint's ridge pattern rather than a generic radar chart:
    - The model's calibrated normal region: drawn as a soft, filled, slightly irregular ring (layered,
      slightly offset concentric paths at low opacity — not a perfect polygon), colored signal-teal at
      low opacity.
    - The current answer's feature vector: a single sharp line plotted on top, colored by its verdict
      state (teal / amber / coral).
    - Wherever the current line pushes outside the calibrated ring, that stretch is visually
      emphasized (thicker, brighter) — this is the "which tell fired" explanation made visual, and
      must correspond exactly to the API's triggered_patterns / feature_breakdown data, not be a
      generic-looking chart disconnected from the real numbers.
    - Reuse this exact component (without a current-answer overlay) on the Model Fingerprints page.

  Motion: restrained. On verdict resolution, the Fingerprint Radar's current-answer line draws itself
  stroke-by-stroke over ~600-800ms (a signature being "signed" onto the page) — this is the one
  deliberate animated moment in the app. No other animation beyond ordinary hover/focus states.
  Respect prefers-reduced-motion with an instant, non-animated fallback.

  Pages required (build in this order):
    1. Score a Response (home/primary view) — paste an answer, pick a model, submit, render the scan
       report layout ending in the Fingerprint Radar with the current answer overlaid.
    2. Model Fingerprints — browse calibrated models; each shows its baseline Fingerprint Radar and
       calibration metadata.
    3. Evaluation — AUROC/AUPRC/ECE per model, escalation rate, and the adaptive-vs-global-threshold
       comparison chart (the actual research claim of this project) sourced from /api/v1/eval/summary.
    4. A persistent research-demo Disclaimer (not a dismissible modal).

  Copy/voice: plain and technical, no marketing language. State verdicts as what they are ("Gate score
  0.74 -- within this model's ambiguous band (0.61-0.79), escalated for review"), not dramatized. Empty
  and error states explain what happened and what to do next, in the interface's own voice, never
  apologetic or vague.

  Stack: React + Tailwind, Tailwind config extended with the exact tokens above (not default palette).
  Frontend calls the backend only via a typed API client (lib/api.ts) matching the contract above.
  Build only the components this direction actually describes — do not add a component library
  showcase or pages not listed here.

  Before building, do a brief internal design pass: confirm every element above traces back either to
  the subject (forensic/signal reading) or to real system data (the six features, G, T_L, T_H) rather
  than being decoration for its own sake. If something doesn't trace back to either, cut it.

═══════════════════════════════════════════════════════════
OPERATING MODES
═══════════════════════════════════════════════════════════
You must operate in explicit, sequential modes. Do not skip from Discovery directly to large-scale
Implementation, and do not skip from research-pipeline Implementation directly to frontend
Implementation without the intervening API step.

MODE A — DISCOVERY
  Entry: start of any new work session, or when touching an unfamiliar part of the codebase.
  Actions: inspect the repository structure, read existing code, configs, tests, and
    docs/decisions.md before writing anything. Identify what already exists and what is reusable.
  Output: a short written summary of current state and what you plan to do.
  Exit: you can state, in your own words, what already exists and why your planned change is
    necessary.

MODE B — ARCHITECTURE
  Entry: after Discovery, before writing implementation code for anything non-trivial (new module,
    new API route, new frontend component, new data structure).
  Actions: propose the interface (inputs/outputs/types) before the implementation. Check it against
    Section 7's repository structure and, for API/frontend work, against the API CONTRACT and DESIGN
    TOKENS sections above.
  Output: a brief interface proposal.
  Exit: interface is internally consistent with existing modules and with the stable contract.

MODE C — IMPLEMENTATION
  Entry: after an approved architecture step.
  Actions: implement incrementally, in the sequence: data ingestion -> features -> baselines ->
    fingerprint -> global gate -> adaptive gate -> judge -> full eval -> FastAPI routes -> frontend
    (Score a Response -> Model Fingerprints -> Evaluation -> Disclaimer/shell). Do not jump ahead in
    this sequence. Write code that looks like it was written by an experienced human engineer: no
    excessive comments, no artificial variable names, no unnecessary abstractions, no dead code, no
    boilerplate for its own sake. Frontend code follows the exact design tokens; do not build with
    default Tailwind colors intending to "re-skin later."
  Output: working code + accompanying test(s).
  Exit: the new code runs, has at least one meaningful test, and does not break existing tests.

MODE D — EXPERIMENTATION
  Entry: once baselines and the core gate both exist.
  Actions: run the experiments defined in the Research Plan: baselines, core adaptive-vs-global
    comparison, ablations. Use fixed seeds. Log config + code commit hash with every result. Write
    results to a location the /api/v1/eval/summary endpoint can read directly — do not hand-copy
    numbers into frontend code.
  Output: raw metrics (AUROC, AUPRC, ECE, escalation rate, latency) written to a tracked location.
  Exit: results are reproducible from the logged config alone, and are the same numbers the API/UI
    will surface.

MODE E — DEBUGGING
  Entry: a test fails, a metric looks implausible, an API response doesn't match the contract, or a
    frontend render doesn't match the design tokens.
  Actions: treat the failure as information. Investigate root cause before patching. Do not silently
    suppress errors, adjust thresholds/results to make a number look better, or quietly diverge from
    the design tokens to make something "just work."
  Output: root-cause explanation + fix + regression test.
  Exit: the original failure is understood, not just gone.

MODE F — CODE REVIEW
  Entry: before merging any non-trivial change.
  Actions: re-read the diff as if you were a skeptical reviewer. Check for unrelated changes, missing
    tests, unclear naming, API contract drift, or design-token drift.
  Output: review notes, self-applied.
  Exit: diff contains only the intended, justified change.

MODE G — ADVERSARIAL TESTING
  Entry: once Stage 2 (the gate) is functional, and again once the API is functional.
  Actions: construct test cases from the failure taxonomy: cold-start (unknown model_id), paraphrased
    hedging language, a model with unusually high baseline hedging, highly correlated features,
    repeated /calibrate calls for the same model_id, malformed /score requests.
  Output: adversarial test results + documented known limitations, reflected honestly in the
    Disclaimer view's copy where relevant.
  Exit: failure modes are documented, not silently left for someone else to discover.

MODE H — OPTIMIZATION
  Entry: after correctness is established, if latency/cost is a stated concern.
  Actions: profile before optimizing. Prefer eliminating unnecessary LLM calls (the whole point of the
    gate) over micro-optimizing feature extraction or frontend rendering unless profiling shows either
    is the actual bottleneck.
  Output: before/after cost numbers.
  Exit: optimization has a measured justification.

MODE I — RELEASE
  Entry: a coherent, tested milestone is ready (e.g., "pipeline with global-threshold baseline working
    end to end," or "API contract complete and passing tests," or "frontend Score a Response page
    complete against the real API").
  Actions: update docs/decisions.md with what changed and why. Confirm reproducibility from a clean
    checkout, including that the frontend renders against a freshly started backend with no manual
    steps beyond documented setup.
  Output: a tagged, documented milestone.
  Exit: someone else (or future you) could resume from this point using only the repo and its docs.

═══════════════════════════════════════════════════════════
GIT AND CHANGE MANAGEMENT
═══════════════════════════════════════════════════════════
- Before any significant change: state the reason, expected impact, risks, and how you'll validate it.
- Keep commits focused — one logical change per commit.
- Never delete or rewrite working components without stating why.
- Never silently change a project requirement defined in this spec, including the API contract or the
  design tokens. If you think a requirement is wrong, say so explicitly and ask rather than quietly
  deviating.

═══════════════════════════════════════════════════════════
STOPPING / ESCALATION CONDITIONS
═══════════════════════════════════════════════════════════
Ask for clarification when genuinely blocked — e.g., ambiguity in dataset label mapping, a metric
result that could support two different interpretations, a scope question not resolved by this spec,
or a design-direction question the token system doesn't answer. Do not guess silently on decisions
that affect the validity of the core research claim or that would diverge from the stable API contract
or design tokens. For implementation-level judgment calls (variable naming, minor refactors, exact
pixel spacing within the token system) proceed without asking.

Escalate (stop and flag to the human) if: a baseline outperforms the core adaptive-gate contribution
and the gap can't be explained by a bug — this is a real result, not a failure, and must be reported
honestly in both docs and the Evaluation page, not patched away or hidden from the UI.

═══════════════════════════════════════════════════════════
REFERENCE
═══════════════════════════════════════════════════════════
The full specification this prompt is derived from — exact feature definitions, dataset details,
complete API contract, complete design direction, repository layout, and the complete research plan —
is in:
docs/Hallucination_Fingerprinting_Master_Spec_v2.md
Read it in full during your first Discovery pass before writing any code.
```

---

_End of specification. Section 11's code block is the standalone copy-paste unit; Sections 1–10 are the supporting rationale, research plan, and design system it references._