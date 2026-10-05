/**
 * ScorePage — Clean Enterprise Response Verification (Lovable / Cloud Platform Style).
 * Displays input controls, telemetry signals, gate score calibration bands, and radar fingerprint.
 */
import { useState, useEffect } from 'react';
import { scoreAnswer, listModels } from '../api';
import FingerprintRadar from '../components/FingerprintRadar';

const VERDICT_CONFIG = {
  LOW_RISK: {
    label: 'Low Hallucination Risk',
    badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    dotClass: 'bg-emerald-500',
    title: 'Verification Passed',
    desc: 'The response aligns tightly with truthful factual density baselines.',
  },
  HIGH_RISK: {
    label: 'High Hallucination Risk',
    badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
    dotClass: 'bg-rose-500',
    title: 'Anomalous Assertions Detected',
    desc: 'High linguistic hedging, unanchored citations, or fabricated specificity identified.',
  },
  AMBIGUOUS: {
    label: 'Ambiguous Edge Case',
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    dotClass: 'bg-amber-500',
    title: 'Escalated to Tie-Breaker',
    desc: 'Telemetry lands in the calibrated uncertainty band between T_L and T_H.',
  },
  RESOLVED_AMBIGUOUS: {
    label: 'Judge Resolved Ambiguity',
    badgeClass: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    dotClass: 'bg-indigo-500',
    title: 'Evaluated by LLM Judge',
    desc: 'Deep reasoning conducted over extracted telemetry features and anomaly patterns.',
  },
};

const FEATURE_NAMES = {
  H: { name: 'Hedge Density', desc: 'Linguistic uncertainty frequency' },
  S: { name: 'Specificity', desc: 'Named entities and concrete anchors' },
  C: { name: 'Citation Vagueness', desc: 'Unverified source gestures' },
  E: { name: 'Evidence Density', desc: 'Checkable assertions per sentence' },
  D: { name: 'Semantic Drift', desc: 'Sentence embedding cosine variance' },
  M: { name: 'Confidence Marker', desc: 'Dogmatic absolute assertions' },
};

export default function ScorePage({ onOpenJudgeSettings }) {
  const [answer, setAnswer] = useState(() => sessionStorage.getItem('hfg_answer') || '');
  const [modelId, setModelId] = useState('');
  const [judgeMode, setJudgeMode] = useState(() => sessionStorage.getItem('hfg_judge_mode') || 'auto');
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(() => {
    try {
      const saved = sessionStorage.getItem('hfg_result');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    listModels()
      .then((data) => {
        // Put universal-llm first as the zero-friction default
        const sorted = [...data].sort((a, b) => {
          if (a.model_id === 'universal-llm') return -1;
          if (b.model_id === 'universal-llm') return 1;
          return a.model_id.localeCompare(b.model_id);
        });
        setModels(sorted);
        const saved = localStorage.getItem('hfg_selected_model');
        const defaultChoice =
          sorted.find((m) => m.model_id === saved) ||
          sorted.find((m) => m.model_id === 'universal-llm') ||
          sorted[0];
        if (defaultChoice) {
          setModelId(defaultChoice.model_id);
        }
      })
      .catch(() => setModels([]));
  }, []);

  function handleModelChange(val) {
    setModelId(val);
    localStorage.setItem('hfg_selected_model', val);
  }

  function handleJudgeModeChange(mode) {
    setJudgeMode(mode);
    sessionStorage.setItem('hfg_judge_mode', mode);
  }

  function handleAnswerChange(val) {
    setAnswer(val);
    sessionStorage.setItem('hfg_answer', val);
  }

  function handleClear() {
    setAnswer('');
    setResult(null);
    setError(null);
    sessionStorage.removeItem('hfg_answer');
    sessionStorage.removeItem('hfg_result');
  }

  function loadSample(type) {
    let text = '';
    if (type === 'factual') {
      text =
        "The Apollo 11 mission landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin explored the lunar surface for over two hours while Michael Collins piloted the command module in orbit.";
    } else if (type === 'hallucinated') {
      text =
        "Studies show that secret ancient pyramids were 100% discovered on Mars in 1994. Clearly, sources indicate that in Sector 7, galactic treaties undoubtedly granted absolute mining rights for 4,500 tons of red crystals.";
    } else if (type === 'ambiguous') {
      text =
        "Research suggests that perhaps advanced ancient civilizations possessed acoustic levitation technology. Some preliminary studies seemingly indicate sound waves could suspend particles, though experts argue empirical evidence remains somewhat debated.";
    }
    setAnswer(text);
    sessionStorage.setItem('hfg_answer', text);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!answer.trim() || !modelId) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await scoreAnswer(answer, modelId, judgeMode);
      setResult(res);
      sessionStorage.setItem('hfg_result', JSON.stringify(res));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8 space-y-6">
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Hallucination Score & Telemetry Scanner
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Submit text to profile latent behavioral drift against calibrated model fingerprints.
          </p>
        </div>

        {/* Quick Judge Mode Selector */}
        <div className="flex items-center gap-2 p-1 bg-white border border-slate-200 rounded-xl shadow-2xs self-start">
          <span className="text-[11px] font-semibold text-slate-500 px-2">Judge Mode:</span>
          {[
            { id: 'auto', label: 'Auto (Coexistence)' },
            { id: 'api', label: 'Cloud API' },
            { id: 'local', label: 'Local Ollama' },
            { id: 'mock', label: 'Fast Mock' },
          ].map((m) => (
            <button
              type="button"
              key={m.id}
              onClick={() => handleJudgeModeChange(m.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                judgeMode === m.id
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              {m.label}
            </button>
          ))}
          <button
            type="button"
            onClick={onOpenJudgeSettings}
            className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md transition-colors cursor-pointer"
            title="Configure API Keys & Engines"
          >
            ⚙️
          </button>
        </div>
      </div>

      {/* Primary Input Card */}
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Target Model Selector */}
            <div className="md:col-span-1">
              <div className="flex items-center justify-between mb-1">
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Target Model Profile
                </label>
                <span className="text-[10px] text-blue-600 font-medium">Auto-Defaulted</span>
              </div>
              <select
                value={modelId}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/10 focus:outline-none transition-all cursor-pointer"
              >
                {models.map((m) => {
                  let display = m.model_id;
                  if (m.model_id === 'universal-llm') display = '🌐 Universal / Any LLM (Default)';
                  else if (m.model_id === 'gpt-4') display = '🧠 OpenAI GPT-4 Profile';
                  else if (m.model_id === 'gpt-3.5-turbo') display = '⚡ OpenAI GPT-3.5 Profile';
                  else if (m.model_id === 'llama-3-8b') display = '🦙 Meta LLaMA-3 (8B) Profile';
                  else if (m.model_id === 'mistral-7b') display = '🌪️ Mistral AI (7B) Profile';
                  return (
                    <option key={m.model_id} value={m.model_id}>
                      {display}
                    </option>
                  );
                })}
              </select>
              <p className="text-[10px] text-slate-400 mt-1 leading-snug">
                Calibrated baseline to compare text against. Leave on <strong>Universal</strong> to test any output without reselecting.
              </p>
            </div>

            {/* Quick Test Presets */}
            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                Benchmark Presets
              </label>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => loadSample('factual')}
                  className="px-3 py-2 text-xs font-medium bg-emerald-50 hover:bg-emerald-100/80 text-emerald-700 border border-emerald-200 rounded-xl transition-colors cursor-pointer flex items-center gap-1.5"
                >
                  <span>✨</span>
                  <span>Factual Apollo 11</span>
                </button>
                <button
                  type="button"
                  onClick={() => loadSample('ambiguous')}
                  className="px-3 py-2 text-xs font-medium bg-amber-50 hover:bg-amber-100/80 text-amber-700 border border-amber-200 rounded-xl transition-colors cursor-pointer flex items-center gap-1.5"
                >
                  <span>⚖️</span>
                  <span>Ambiguous Levitation</span>
                </button>
                <button
                  type="button"
                  onClick={() => loadSample('hallucinated')}
                  className="px-3 py-2 text-xs font-medium bg-rose-50 hover:bg-rose-100/80 text-rose-700 border border-rose-200 rounded-xl transition-colors cursor-pointer flex items-center gap-1.5"
                >
                  <span>⚠️</span>
                  <span>Mars Pyramids (Hallucinated)</span>
                </button>
              </div>
            </div>
          </div>

          {/* Response Textarea */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Response Text to Verify
              </label>
              <span className="text-[11px] text-slate-400 font-mono">
                {answer.length.toLocaleString()} characters
              </span>
            </div>
            <textarea
              value={answer}
              onChange={(e) => handleAnswerChange(e.target.value)}
              rows={4}
              placeholder="Paste generated model output here or pick a benchmark preset above..."
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3.5 text-xs text-slate-800 placeholder-slate-400 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/10 focus:outline-none transition-all leading-relaxed"
            />
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-1">
            <span className="text-[11px] text-slate-400">
              Evaluates 6 mathematical features in milliseconds • Escalates only when ambiguous
            </span>
            <div className="flex items-center gap-2">
              {(answer || result) && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-all cursor-pointer"
                >
                  Clear
                </button>
              )}
              <button
                type="submit"
                disabled={loading || !answer.trim() || !modelId}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-xl shadow-xs transition-all flex items-center gap-2 cursor-pointer"
              >
                {loading ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Computing Telemetry...</span>
                  </>
                ) : (
                  <>
                    <span>Run Verification</span>
                    <span>→</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-start gap-2">
            <span className="text-sm">⚠️</span>
            <div>
              <p className="font-semibold">Verification Error</p>
              <p className="mt-0.5">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Results Scan Report */}
      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-200">
          {/* Verdict Banner Card */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span
                  className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border ${
                    VERDICT_CONFIG[result.verdict]?.badgeClass || 'bg-slate-100 text-slate-700'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      VERDICT_CONFIG[result.verdict]?.dotClass || 'bg-slate-400'
                    }`}
                  />
                  {VERDICT_CONFIG[result.verdict]?.label || result.verdict}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  resolved via:{' '}
                  <strong className="text-slate-700 capitalize">{result.resolved_by.replace('_', ' ')}</strong>
                </span>
              </div>

              <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
                <span>Model: <strong className="text-slate-800">{result.model_id}</strong></span>
                <span>•</span>
                <span>Fingerprint: <strong className="text-slate-800">{result.fingerprint_version}</strong></span>
              </div>
            </div>

            {/* Gate Score Calibration Track */}
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-emerald-700 font-semibold">
                  T_Low = {result.thresholds.t_low.toFixed(3)}
                </span>
                <span className="text-slate-900 font-bold bg-white px-2.5 py-0.5 rounded-md border border-slate-200 shadow-2xs">
                  Gate Score G = {result.gate_score.toFixed(4)}
                </span>
                <span className="text-rose-700 font-semibold">
                  T_High = {result.thresholds.t_high.toFixed(3)}
                </span>
              </div>

              {/* Progress Gauge */}
              <div className="relative h-3.5 bg-slate-200 rounded-full overflow-hidden">
                {/* Safe zone */}
                <div
                  className="absolute inset-y-0 left-0 bg-emerald-300/80"
                  style={{ width: `${result.thresholds.t_low * 100}%` }}
                />
                {/* Ambiguous zone */}
                <div
                  className="absolute inset-y-0 bg-amber-300/80"
                  style={{
                    left: `${result.thresholds.t_low * 100}%`,
                    width: `${Math.max((result.thresholds.t_high - result.thresholds.t_low) * 100, 0)}%`,
                  }}
                />
                {/* High risk zone */}
                <div
                  className="absolute inset-y-0 bg-rose-300/80"
                  style={{
                    left: `${result.thresholds.t_high * 100}%`,
                    right: 0,
                  }}
                />
                {/* Indicator marker */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-slate-950 shadow-md ring-2 ring-white"
                  style={{ left: `${Math.min(result.gate_score * 100, 100)}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[10px] text-slate-400 font-medium px-1">
                <span>0.0 (Safe Factual)</span>
                <span>Ambiguity Band (Judge Escalation)</span>
                <span>1.0 (Anomalous)</span>
              </div>
            </div>
          </div>

          {/* Details Grid: Left (Telemetry + Patterns + Explanation), Right (Fingerprint Radar) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column */}
            <div className="lg:col-span-7 space-y-4">
              {/* Telemetry Breakdown */}
              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-3">
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Mathematical Telemetry Signals
                </h3>
                <div className="space-y-2.5">
                  {Object.entries(result.feature_breakdown).map(([key, val]) => (
                    <div key={key} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-1.5">
                          <span className="w-5 h-5 flex items-center justify-center rounded font-mono font-bold bg-slate-100 text-slate-800 text-[10px]">
                            {key}
                          </span>
                          <span className="font-medium text-slate-700">
                            {FEATURE_NAMES[key]?.name || key}
                          </span>
                          <span className="text-[10px] text-slate-400 hidden sm:inline">
                            ({FEATURE_NAMES[key]?.desc})
                          </span>
                        </div>
                        <span className="font-mono font-semibold text-slate-800">
                          {val.toFixed(4)}
                        </span>
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-600 rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(val * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Triggered Symbolic Anomaly Patterns */}
              {result.triggered_patterns.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-3">
                  <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <span>⚡ Activated Neuro-Symbolic Anomaly Rules</span>
                  </h3>
                  <div className="space-y-2">
                    {result.triggered_patterns.map((p, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs"
                      >
                        <div className="space-y-0.5">
                          <span className="font-semibold text-slate-900 font-mono">
                            {p.name.replace(/_/g, ' ')}
                          </span>
                          <div className="text-[11px] text-slate-500">
                            Features involved: [
                            <span className="font-mono font-medium text-blue-600">
                              {p.features_involved.join(', ')}
                            </span>
                            ]
                          </div>
                        </div>
                        <div className="text-right">
                          <span className="px-2 py-0.5 rounded-full text-[11px] font-mono font-bold bg-amber-100 text-amber-800 border border-amber-200">
                            Activation: {p.strength.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Natural Language Explanation */}
              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Attribution & Explanation
                  </h3>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                    Source: {result.explanation_source}
                  </span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200/70 font-sans">
                  {result.explanation}
                </p>
              </div>
            </div>

            {/* Right Column: Fingerprint Radar Visualization */}
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-2xl shadow-sm p-5 flex flex-col items-center justify-center space-y-3">
              <div className="w-full text-center pb-2 border-b border-slate-100">
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Fingerprint Radar Signature
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Overlaid against model calibrated baseline centroid
                </p>
              </div>

              <FingerprintRadar
                featureBreakdown={result.feature_breakdown}
                verdict={result.verdict}
                size={300}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
