/**
 * Evaluation — the research view (spec Section 5.13 view 3).
 *
 * Shows AUROC/AUPRC/ECE per model, escalation rate, and the
 * adaptive-vs-global comparison — the actual research claim.
 */
import { useState, useEffect } from 'react';
import { getEvalSummary } from '../api';

const METRIC_LABELS = {
  auroc: 'AUROC',
  auprc: 'AUPRC',
  ece: 'ECE',
  escalation_rate: 'Escalation Rate',
};

function MetricBar({ label, value, max = 1.0, color = 'signal-teal' }) {
  const pct = Math.min((value / max) * 100, 100);
  const isNaN_ = isNaN(value);
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-mono text-paper-400 w-28">{label}</span>
      <div className="flex-1 h-2 bg-graphite-600 rounded-full overflow-hidden">
        {!isNaN_ && (
          <div
            className={`h-full bg-${color} rounded-full`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span className="text-xs font-mono text-paper-100 w-16 text-right">
        {isNaN_ ? 'N/A' : value.toFixed(4)}
      </span>
    </div>
  );
}

export default function EvalPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getEvalSummary()
      .then(setData)
      .catch(() => setData({ experiments: [] }))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-3xl font-display font-semibold text-paper-100">
        Evaluation
      </h1>
      <p className="mt-2 text-sm text-paper-400">
        Research evaluation: AUROC, AUPRC (primary metric), ECE, escalation rate,
        and baseline comparison. This page makes the Section 6 core experiment
        legible to a viewer.
      </p>

      {loading ? (
        <p className="mt-8 text-paper-400">Loading evaluation results...</p>
      ) : !data?.experiments?.length ? (
        <div className="mt-8 bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-8 text-center">
          <p className="text-paper-400">No evaluation results yet.</p>
          <p className="text-xs text-paper-400 mt-2">
            Run the evaluation harness to generate baseline comparison data.
          </p>
        </div>
      ) : (
        <div className="mt-8 space-y-6">
          {/* Baseline comparison table */}
          <div className="bg-graphite-800 border border-graphite-600 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-graphite-600">
                  <th className="px-4 py-3 text-left text-paper-400 font-normal">
                    Experiment
                  </th>
                  <th className="px-4 py-3 text-right text-paper-400 font-normal font-mono">
                    AUROC
                  </th>
                  <th className="px-4 py-3 text-right text-paper-400 font-normal font-mono">
                    AUPRC
                  </th>
                  <th className="px-4 py-3 text-right text-paper-400 font-normal font-mono">
                    ECE
                  </th>
                  <th className="px-4 py-3 text-right text-paper-400 font-normal font-mono">
                    Esc. Rate
                  </th>
                  <th className="px-4 py-3 text-right text-paper-400 font-normal">
                    N
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.experiments.map((exp, i) => (
                  <tr
                    key={i}
                    className="border-b border-graphite-600 last:border-0 hover:bg-graphite-950/30"
                  >
                    <td className="px-4 py-3 font-mono text-paper-100">
                      {exp.predictor || exp.experiment}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-paper-100">
                      {isNaN(exp.metrics.auroc) ? 'N/A' : exp.metrics.auroc.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-signal-teal">
                      {isNaN(exp.metrics.auprc) ? 'N/A' : exp.metrics.auprc.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-paper-100">
                      {exp.metrics.ece.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-signal-amber">
                      {(exp.metrics.escalation_rate * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-paper-400">
                      {exp.n_records}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-experiment detail cards */}
          {data.experiments.map((exp, i) => (
            <div
              key={i}
              className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-5"
            >
              <div className="flex items-baseline justify-between mb-3">
                <h3 className="font-mono text-paper-100">
                  {exp.predictor || exp.experiment}
                </h3>
                <span className="text-xs text-paper-400 font-mono">
                  {exp.n_positive} pos / {exp.n_negative} neg
                </span>
              </div>
              <div className="space-y-2">
                <MetricBar label="AUROC" value={exp.metrics.auroc} color="signal-teal" />
                <MetricBar label="AUPRC" value={exp.metrics.auprc} color="signal-teal" />
                <MetricBar label="ECE" value={exp.metrics.ece} color="signal-amber" />
                <MetricBar
                  label="Escalation"
                  value={exp.metrics.escalation_rate}
                  color="signal-amber"
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
