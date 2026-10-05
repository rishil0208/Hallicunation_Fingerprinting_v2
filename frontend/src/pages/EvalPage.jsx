/**
 * EvalPage — Clean Enterprise Benchmark Evaluation View.
 * Displays AUROC, AUPRC (primary metric), ECE, escalation rates, and baseline comparisons.
 */
import { useState, useEffect } from 'react';
import { getEvalSummary } from '../api';

function MetricBar({ label, value, max = 1.0, color = 'bg-blue-600' }) {
  const pct = Math.min((value / max) * 100, 100);
  const isNaN_ = isNaN(value);
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-mono font-medium text-slate-500 w-28">{label}</span>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200/50">
        {!isNaN_ && (
          <div
            className={`h-full ${color} rounded-full transition-all duration-500`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span className="text-xs font-mono font-bold text-slate-800 w-16 text-right">
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
    <div className="mx-auto max-w-5xl px-6 py-8 space-y-6">
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Research Evaluation Benchmarks
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          Comparative empirical metrics: AUROC, AUPRC (primary metric), ECE calibration error, and escalation rates.
        </p>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-400 text-xs">
          Loading evaluation metrics...
        </div>
      ) : !data?.experiments?.length ? (
        <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center shadow-xs">
          <p className="text-sm font-semibold text-slate-700">No evaluation runs recorded yet.</p>
          <p className="text-xs text-slate-400 mt-1">
            Execute the benchmark harness via <code className="font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded">python eval/run_experiment.py</code>
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Comparison Table */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
              <h2 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Benchmark Comparison Summary
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80 text-slate-600">
                    <th className="px-6 py-3 text-left font-semibold">Experiment / Baseline</th>
                    <th className="px-4 py-3 text-right font-semibold font-mono">AUROC</th>
                    <th className="px-4 py-3 text-right font-semibold font-mono text-blue-600">AUPRC (Primary)</th>
                    <th className="px-4 py-3 text-right font-semibold font-mono">ECE</th>
                    <th className="px-4 py-3 text-right font-semibold font-mono text-amber-600">Esc. Rate</th>
                    <th className="px-6 py-3 text-right font-semibold font-mono">Samples</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.experiments.map((exp, i) => (
                    <tr key={i} className="hover:bg-slate-50/70 transition-colors">
                      <td className="px-6 py-3.5 font-mono font-medium text-slate-900">
                        {exp.predictor || exp.experiment}
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono text-slate-700">
                        {isNaN(exp.metrics.auroc) ? 'N/A' : exp.metrics.auroc.toFixed(4)}
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono font-bold text-blue-600 bg-blue-50/30">
                        {isNaN(exp.metrics.auprc) ? 'N/A' : exp.metrics.auprc.toFixed(4)}
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono text-slate-700">
                        {exp.metrics.ece.toFixed(4)}
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono font-semibold text-amber-600">
                        {(exp.metrics.escalation_rate * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-500">
                        {exp.n_records.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-Experiment Detail Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.experiments.map((exp, i) => (
              <div
                key={i}
                className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-3"
              >
                <div className="flex items-baseline justify-between pb-2 border-b border-slate-100">
                  <h3 className="font-mono text-sm font-bold text-slate-900">
                    {exp.predictor || exp.experiment}
                  </h3>
                  <span className="text-[11px] font-mono text-slate-400">
                    {exp.n_positive} positive / {exp.n_negative} negative
                  </span>
                </div>
                <div className="space-y-2.5 pt-1">
                  <MetricBar label="AUROC" value={exp.metrics.auroc} color="bg-blue-500" />
                  <MetricBar label="AUPRC (Primary)" value={exp.metrics.auprc} color="bg-blue-600" />
                  <MetricBar label="ECE Calibration" value={exp.metrics.ece} color="bg-amber-500" />
                  <MetricBar
                    label="Escalation Rate"
                    value={exp.metrics.escalation_rate}
                    color="bg-amber-600"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
