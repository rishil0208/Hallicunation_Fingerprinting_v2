/**
 * ModelsPage — Clean Enterprise Model Fingerprints Viewer.
 * Inspect calibrated baseline centroids, thresholds, and weights per model.
 */
import { useState, useEffect } from 'react';
import { listModels, getFingerprint } from '../api';
import FingerprintRadar from '../components/FingerprintRadar';

export default function ModelsPage() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [fingerprint, setFingerprint] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        setLoading(false);
        if (m.length > 0) {
          selectModel(m[0].model_id);
        }
      })
      .catch(() => setLoading(false));
  }, []);

  async function selectModel(modelId) {
    setSelected(modelId);
    try {
      const fp = await getFingerprint(modelId);
      setFingerprint(fp);
    } catch {
      setFingerprint(null);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8 space-y-6">
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Calibrated Model Fingerprints
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          Browse learned baseline centroids, decision bands ($T_L, T_H$), and neuro-symbolic weights ($w_i$).
        </p>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-400 text-xs">
          Loading calibrated profiles...
        </div>
      ) : models.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center shadow-xs">
          <p className="text-sm font-semibold text-slate-700">No calibrated models found.</p>
          <p className="text-xs text-slate-400 mt-1">
            Run calibration via <code className="font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded">POST /api/v1/models/{'{id}'}/calibrate</code>
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Models Navigation List */}
          <div className="md:col-span-4 space-y-2">
            <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Available Models ({models.length})
            </h2>
            {models.map((m) => {
              const isSelected = selected === m.model_id;
              return (
                <button
                  key={m.model_id}
                  onClick={() => selectModel(m.model_id)}
                  className={`w-full text-left p-4 rounded-xl border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-blue-50/60 border-blue-500 ring-2 ring-blue-500/10 shadow-xs'
                      : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-sm text-slate-900 font-mono">
                      {m.model_id}
                    </span>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                        isSelected ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      v{m.fingerprint_version}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1.5 flex items-center gap-2">
                    <span>{m.calibration_dataset_size.toLocaleString()} samples</span>
                    <span>•</span>
                    <span className="truncate">Calibrated profile</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Fingerprint Profile Details */}
          <div className="md:col-span-8">
            {fingerprint ? (
              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-6">
                <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                  <div>
                    <h2 className="text-lg font-bold text-slate-900 font-mono">
                      {fingerprint.model_id}
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Baseline signature & learned calibration intervals
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-200">
                    Version {fingerprint.version}
                  </span>
                </div>

                {/* Radar Plot */}
                <div className="flex flex-col items-center justify-center p-4 bg-slate-50/50 rounded-xl border border-slate-100">
                  <FingerprintRadar
                    fingerprint={fingerprint}
                    size={280}
                  />
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-medium text-slate-500 uppercase">T_Low (Pass)</span>
                    <p className="text-base font-mono font-bold text-emerald-600 mt-0.5">
                      {fingerprint.t_low?.toFixed(3) ?? 'N/A'}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-medium text-slate-500 uppercase">T_High (Flag)</span>
                    <p className="text-base font-mono font-bold text-rose-600 mt-0.5">
                      {fingerprint.t_high?.toFixed(3) ?? 'N/A'}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-medium text-slate-500 uppercase">Dataset Size</span>
                    <p className="text-base font-mono font-bold text-slate-800 mt-0.5">
                      {fingerprint.calibration_dataset_size.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-medium text-slate-500 uppercase">Ambiguity Band</span>
                    <p className="text-base font-mono font-bold text-amber-600 mt-0.5">
                      {fingerprint.t_low && fingerprint.t_high
                        ? (fingerprint.t_high - fingerprint.t_low).toFixed(3)
                        : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Learned Rule Weights */}
                {fingerprint.w_i && (
                  <div className="space-y-2">
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Learned Anomaly Weights (w_i)
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {Object.entries(fingerprint.w_i).map(([rule, weight]) => (
                        <div
                          key={rule}
                          className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between"
                        >
                          <span className="text-[11px] font-mono font-medium text-slate-600 truncate">
                            {rule.replace(/_/g, ' ')}
                          </span>
                          <span className="text-sm font-mono font-bold text-blue-600 mt-1">
                            {weight.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center text-slate-400 text-xs">
                Select a model on the left to inspect its calibrated fingerprint.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
