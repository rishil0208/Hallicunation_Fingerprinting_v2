/**
 * Score a Response — primary/home view (spec Section 5.13 view 1).
 *
 * Scan-report layout: input → extracted telemetry → gate verdict
 * with score plotted between T_L and T_H → (if escalated) judge explanation.
 */
import { useState, useEffect } from 'react';
import { scoreAnswer, listModels } from '../api';
import FingerprintRadar from '../components/FingerprintRadar';

const VERDICT_STYLES = {
  LOW_RISK: 'verdict-low-risk',
  HIGH_RISK: 'verdict-high-risk',
  AMBIGUOUS: 'verdict-ambiguous',
  RESOLVED_AMBIGUOUS: 'verdict-resolved',
};

export default function ScorePage() {
  const [answer, setAnswer] = useState('');
  const [modelId, setModelId] = useState('');
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!answer.trim() || !modelId) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await scoreAnswer(answer, modelId);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 px-6 py-10">
      <div>
        <h1 className="text-3xl font-display font-semibold text-paper-100">
          Score a Response
        </h1>
        <p className="mt-2 text-sm text-paper-400">
          Paste a model response, select the originating model, and run the
          three-stage detection pipeline.
        </p>
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-paper-400 mb-1">Model</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full bg-graphite-800 border border-graphite-600 rounded px-3 py-2 text-paper-100 focus:border-signal-teal focus:outline-none"
          >
            <option value="">Select a calibrated model</option>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id} (v{m.fingerprint_version})
              </option>
            ))}
          </select>
          {models.length === 0 && (
            <p className="text-xs text-paper-400 mt-1">
              No calibrated models. Calibrate a model first via the API or Models page.
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm text-paper-400 mb-1">
            Response to analyze
          </label>
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            rows={6}
            placeholder="Paste the model's response here..."
            className="w-full bg-graphite-800 border border-graphite-600 rounded px-3 py-2 text-paper-100 font-mono text-sm resize-y focus:border-signal-teal focus:outline-none"
          />
        </div>

        <button
          type="submit"
          disabled={loading || !answer.trim() || !modelId}
          className="bg-signal-teal text-graphite-950 px-6 py-2 rounded font-medium text-sm hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </form>

      {error && (
        <div className="bg-graphite-800 border border-signal-coral/30 rounded px-4 py-3 text-sm text-signal-coral">
          {error}
        </div>
      )}

      {/* Scan report */}
      {result && (
        <div className="space-y-6">
          {/* Verdict header */}
          <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5">
            <div className="flex items-baseline justify-between">
              <h2 className={`text-2xl font-display font-semibold ${VERDICT_STYLES[result.verdict]}`}>
                {result.verdict.replace(/_/g, ' ')}
              </h2>
              <span className="font-mono text-sm text-paper-400">
                resolved by: {result.resolved_by}
              </span>
            </div>

            {/* Gate score bar */}
            <div className="mt-4">
              <div className="flex items-center justify-between text-xs font-mono text-paper-400 mb-1">
                <span>T_L = {result.thresholds.t_low.toFixed(3)}</span>
                <span>G = {result.gate_score.toFixed(4)}</span>
                <span>T_H = {result.thresholds.t_high.toFixed(3)}</span>
              </div>
              <div className="relative h-3 bg-graphite-600 rounded-full overflow-hidden">
                {/* Low risk zone */}
                <div
                  className="absolute h-full bg-signal-teal/20"
                  style={{ left: 0, width: `${result.thresholds.t_low * 100}%` }}
                />
                {/* Ambiguous zone */}
                <div
                  className="absolute h-full bg-signal-amber/20"
                  style={{
                    left: `${result.thresholds.t_low * 100}%`,
                    width: `${(result.thresholds.t_high - result.thresholds.t_low) * 100}%`,
                  }}
                />
                {/* High risk zone */}
                <div
                  className="absolute h-full bg-signal-coral/20"
                  style={{
                    left: `${result.thresholds.t_high * 100}%`,
                    width: `${(1 - result.thresholds.t_high) * 100}%`,
                  }}
                />
                {/* Score marker */}
                <div
                  className="absolute top-0 h-full w-0.5 bg-paper-100"
                  style={{ left: `${Math.min(result.gate_score * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Feature telemetry */}
          <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5">
            <h3 className="text-sm font-display text-paper-400 mb-3">
              Extracted Telemetry
            </h3>
            <div className="space-y-2">
              {Object.entries(result.feature_breakdown).map(([key, val]) => (
                <div key={key} className="flex items-center gap-3">
                  <span className="font-mono text-xs text-paper-400 w-4">{key}</span>
                  <div className="flex-1 h-2 bg-graphite-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-signal-teal rounded-full transition-all"
                      style={{ width: `${Math.min(val * 100, 100)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs text-paper-100 w-14 text-right">
                    {val.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Triggered patterns */}
          {result.triggered_patterns.length > 0 && (
            <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5">
              <h3 className="text-sm font-display text-paper-400 mb-3">
                Triggered Patterns
              </h3>
              <div className="space-y-2">
                {result.triggered_patterns.map((p, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="font-mono text-paper-100">{p.name}</span>
                    <span className="text-paper-400">
                      [{p.features_involved.join(', ')}] ·{' '}
                      <span className="font-mono text-signal-amber">
                        {p.strength.toFixed(3)}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Explanation */}
          <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5">
            <h3 className="text-sm font-display text-paper-400 mb-2">
              Explanation
              <span className="ml-2 text-xs font-mono">
                ({result.explanation_source})
              </span>
            </h3>
            <p className="text-sm text-paper-100 leading-relaxed">
              {result.explanation}
            </p>
          </div>

          {/* Fingerprint Radar */}
          <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5 flex flex-col items-center">
            <h3 className="text-sm font-display text-paper-400 mb-4">
              Fingerprint Radar
            </h3>
            <FingerprintRadar
              featureBreakdown={result.feature_breakdown}
              verdict={result.verdict}
              size={320}
            />
          </div>

          {/* Metadata */}
          <div className="text-xs font-mono text-paper-400 flex gap-4">
            <span>model: {result.model_id}</span>
            <span>fingerprint: {result.fingerprint_version}</span>
          </div>
        </div>
      )}
    </div>
  );
}
