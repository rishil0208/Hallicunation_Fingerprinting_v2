import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import ScorePage from './pages/ScorePage';
import ModelsPage from './pages/ModelsPage';
import EvalPage from './pages/EvalPage';
import Disclaimer from './components/Disclaimer';
import JudgeSettingsModal from './components/JudgeSettingsModal';
import { getJudgeStatus } from './api';

const NAV_ITEMS = [
  { to: '/', label: 'Score Response' },
  { to: '/models', label: 'Model Fingerprints' },
  { to: '/evaluation', label: 'Evaluation Benchmarks' },
];

export default function App() {
  const [judgeModalOpen, setJudgeModalOpen] = useState(false);
  const [judgeStatus, setJudgeStatus] = useState(null);

  useEffect(() => {
    getJudgeStatus()
      .then(setJudgeStatus)
      .catch(() => {});
  }, []);

  function getJudgeBadge() {
    if (!judgeStatus) return '⚖️ Judge: Loading...';
    if (judgeStatus.active_mode === 'api') {
      return judgeStatus.api_key_configured
        ? '⚡ Cloud API (Gemini Active)'
        : '⚠️ Cloud API (Key Needed)';
    }
    if (judgeStatus.active_mode === 'local') {
      return judgeStatus.local_ollama_available
        ? '💻 Local LLM (Qwen Active)'
        : '⚠️ Local LLM (Ollama Offline)';
    }
    if (judgeStatus.active_mode === 'mock') {
      return '🛡️ Symbolic Fast Mode';
    }
    // Auto
    if (judgeStatus.api_key_configured) return '⚡ Smart Auto (API Key)';
    if (judgeStatus.local_ollama_available) return '💻 Smart Auto (Ollama)';
    return '🛡️ Smart Auto (Neuro-Symbolic)';
  }

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-slate-800 font-sans selection:bg-blue-100 selection:text-blue-900">
        {/* Sticky Header */}
        <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-xs">
          <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-6">
            {/* Brand Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-sm shadow-blue-500/20">
                <span className="text-base font-extrabold tracking-wider">H</span>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 text-sm tracking-tight font-display">
                    HFG GATE
                  </span>
                  <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                    v2.0
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 hidden sm:block">
                  Calibrated Neuro-Symbolic Verification
                </p>
              </div>
            </div>

            {/* Navigation tabs */}
            <nav className="flex items-center gap-1.5 p-1 bg-slate-100/80 rounded-xl border border-slate-200/80">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                      isActive
                        ? 'bg-white text-blue-600 shadow-xs border border-slate-200/60'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>

            {/* Right: Judge Status & Settings Button */}
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={() => setJudgeModalOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 hover:border-slate-300 text-xs font-medium text-slate-700 shadow-2xs transition-all cursor-pointer group"
                title="Configure LLM Judge / API Key"
              >
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-mono text-[11px]">{getJudgeBadge()}</span>
                <span className="text-[10px] text-blue-600 group-hover:translate-x-0.5 transition-transform font-bold">
                  ⚙️
                </span>
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<ScorePage onOpenJudgeSettings={() => setJudgeModalOpen(true)} />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/evaluation" element={<EvalPage />} />
          </Routes>
        </main>

        {/* Global Footer & Disclaimer */}
        <Disclaimer />

        {/* Judge Settings & API Key Modal */}
        <JudgeSettingsModal
          isOpen={judgeModalOpen}
          onClose={() => setJudgeModalOpen(false)}
          onConfigured={(newStatus) => setJudgeStatus(newStatus)}
        />
      </div>
    </BrowserRouter>
  );
}
