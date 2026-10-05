import { useState, useEffect } from 'react';
import { getJudgeStatus, configureJudge } from '../api';

export default function JudgeSettingsModal({ isOpen, onClose, onConfigured }) {
  const [status, setStatus] = useState(null);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [selectedMode, setSelectedMode] = useState('auto');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadStatus();
    }
  }, [isOpen]);

  async function loadStatus() {
    setLoading(true);
    setMessage(null);
    try {
      const data = await getJudgeStatus();
      setStatus(data);
      setSelectedMode(data.active_mode || 'auto');
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setMessage(null);
    try {
      const updated = await configureJudge({
        apiKey: apiKeyInput.trim() ? apiKeyInput.trim() : null,
        judgeMode: selectedMode,
      });
      setStatus(updated);
      setApiKeyInput('');
      setMessage({ type: 'success', text: 'Judge settings updated successfully!' });
      if (onConfigured) onConfigured(updated);
      setTimeout(() => {
        onClose();
      }, 1200);
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Failed to save configuration.' });
    } finally {
      setSaving(false);
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="relative w-full max-w-xl bg-white border border-slate-200 rounded-2xl shadow-xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div className="flex items-center gap-2.5">
            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-50 text-blue-600 font-bold text-sm">
              ⚖️
            </span>
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Judge Engine & API Configuration
              </h2>
              <p className="text-xs text-slate-500">
                Coexistence of Cloud API & Local LLM with token-saving guardrails
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 rounded-lg p-1 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSave} className="p-6 space-y-5">
          {message && (
            <div
              className={`p-3 rounded-xl text-xs font-medium border ${
                message.type === 'success'
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-rose-50 text-rose-800 border-rose-200'
              }`}
            >
              {message.text}
            </div>
          )}

          {/* Mode Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Resolution Strategy
            </label>
            <div className="grid grid-cols-2 gap-2.5">
              {[
                {
                  id: 'auto',
                  title: 'Smart Coexistence',
                  badge: 'Recommended',
                  desc: 'Uses Cloud API if key is present, local Ollama if running, else neuro-symbolic fallback.',
                },
                {
                  id: 'api',
                  title: 'Cloud API (Gemini)',
                  badge: 'Fast & Accurate',
                  desc: 'Uses your Gemini API key strictly for ambiguous edge cases.',
                },
                {
                  id: 'local',
                  title: 'Local LLM (Ollama)',
                  badge: '100% Offline',
                  desc: 'Queries Qwen 2.5 via local Ollama instance on port 11434.',
                },
                {
                  id: 'mock',
                  title: 'Symbolic Fast Mode',
                  badge: 'Zero Latency',
                  desc: 'Uses deterministic rule-grounded logic. 0 cost, no API keys needed.',
                },
              ].map((opt) => (
                <button
                  type="button"
                  key={opt.id}
                  onClick={() => setSelectedMode(opt.id)}
                  className={`text-left p-3 rounded-xl border transition-all ${
                    selectedMode === opt.id
                      ? 'border-blue-600 bg-blue-50/60 ring-2 ring-blue-500/10'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <span className="text-xs font-semibold text-slate-900">{opt.title}</span>
                    <span
                      className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
                        selectedMode === opt.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {opt.badge}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-relaxed">{opt.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Gemini API Key Section */}
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                  <span>Google Gemini API Key</span>
                  {status?.api_key_configured ? (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded-full">
                      ✓ Active ({status.api_key_masked})
                    </span>
                  ) : (
                    <span className="text-[11px] font-normal text-amber-700 bg-amber-100/70 px-2 py-0.5 rounded-full">
                      Not Configured
                    </span>
                  )}
                </label>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  You can provide your key now or later. Stored securely and masked.
                </p>
              </div>
            </div>

            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder={
                  status?.api_key_configured
                    ? '•••••••••••••••••••••••• (Leave blank to keep existing key)'
                    : 'Paste your Gemini API key (AIzaSy...)'
                }
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 pr-16"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-slate-500 hover:text-slate-800 px-2 py-1 rounded"
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          {/* Guardrails Callout */}
          <div className="p-3.5 rounded-xl bg-blue-50/50 border border-blue-100 text-[11px] space-y-1.5">
            <div className="font-semibold text-blue-900 flex items-center gap-1.5">
              <span>🛡️ Token & Cost Protection Guardrails</span>
            </div>
            <ul className="text-blue-800/80 space-y-1 list-disc list-inside">
              <li><strong>Zero-Waste Gate:</strong> Key is only invoked when score is strictly ambiguous (T_L ≤ G ≤ T_H). Safe/obvious queries cost 0 API calls.</li>
              <li><strong>Token Capped:</strong> Capped at 256 output tokens and 0.1 temperature to prevent runaway spend.</li>
              <li><strong>In-Memory Cache:</strong> Identical requests are answered instantly without hitting the API.</li>
              <li><strong>Privacy:</strong> Key is never logged to files or console.</li>
            </ul>
          </div>

          {/* Footer actions */}
          <div className="pt-2 flex items-center justify-between border-t border-slate-100">
            <div className="text-[11px] text-slate-400">
              Model: <span className="font-mono text-slate-600">{status?.gemini_model || 'gemini-2.0-flash'}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3.5 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="px-4 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-lg shadow-sm transition-all disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Apply & Save'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
