/**
 * Model Fingerprints — browse calibrated models (spec Section 5.13 view 2).
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
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-3xl font-display font-semibold text-paper-100">
        Model Fingerprints
      </h1>
      <p className="mt-2 text-sm text-paper-400">
        Browse calibrated models and view each one's baseline Fingerprint Radar
        and calibration metadata.
      </p>

      {loading ? (
        <p className="mt-8 text-paper-400">Loading models...</p>
      ) : models.length === 0 ? (
        <div className="mt-8 bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-8 text-center">
          <p className="text-paper-400">No calibrated models available.</p>
          <p className="text-xs text-paper-400 mt-2">
            Use <code className="font-mono text-signal-teal">POST /api/v1/models/{'<model_id>'}/calibrate</code> to calibrate a model.
          </p>
        </div>
      ) : (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Model list */}
          <div className="space-y-2">
            {models.map((m) => (
              <button
                key={m.model_id}
                onClick={() => selectModel(m.model_id)}
                className={`w-full text-left px-4 py-3 rounded border transition-colors ${
                  selected === m.model_id
                    ? 'bg-graphite-800 border-signal-teal text-paper-100'
                    : 'bg-graphite-800 border-graphite-600 text-paper-400 hover:border-paper-400'
                }`}
              >
                <div className="font-mono text-sm">{m.model_id}</div>
                <div className="text-xs mt-1">
                  v{m.fingerprint_version} · {m.calibration_dataset_size} samples
                </div>
              </button>
            ))}
          </div>

          {/* Fingerprint detail */}
          <div className="md:col-span-2">
            {fingerprint ? (
              <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-6 space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="font-display text-xl text-paper-100">
                    {fingerprint.model_id}
                  </h2>
                  <span className="font-mono text-xs text-paper-400">
                    v{fingerprint.version}
                  </span>
                </div>

                <FingerprintRadar
                  fingerprint={fingerprint}
                  size={300}
                />

                {/* Calibration metadata */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-paper-400">T_L</span>
                    <p className="font-mono text-signal-teal">
                      {fingerprint.t_low?.toFixed(4) ?? 'uncalibrated'}
                    </p>
                  </div>
                  <div>
                    <span className="text-paper-400">T_H</span>
                    <p className="font-mono text-signal-coral">
                      {fingerprint.t_high?.toFixed(4) ?? 'uncalibrated'}
                    </p>
                  </div>
                  <div>
                    <span className="text-paper-400">Dataset size</span>
                    <p className="font-mono text-paper-100">
                      {fingerprint.calibration_dataset_size}
                    </p>
                  </div>
                  <div>
                    <span className="text-paper-400">Last calibrated</span>
                    <p className="font-mono text-paper-100 text-xs">
                      {fingerprint.last_calibrated_at ?? 'unknown'}
                    </p>
                  </div>
                </div>

                {/* Rule weights */}
                {fingerprint.w_i && (
                  <div>
                    <h3 className="text-sm text-paper-400 mb-2">Rule Weights (w_i)</h3>
                    <div className="space-y-1">
                      {Object.entries(fingerprint.w_i).map(([rule, weight]) => (
                        <div key={rule} className="flex justify-between text-xs">
                          <span className="font-mono text-paper-100">{rule}</span>
                          <span className="font-mono text-signal-amber">
                            {weight.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : selected ? (
              <p className="text-paper-400">Loading fingerprint...</p>
            ) : (
              <div className="bg-graphite-800 border border-graphite-600 rounded-lg px-6 py-12 text-center text-paper-400">
                Select a model to view its fingerprint.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
