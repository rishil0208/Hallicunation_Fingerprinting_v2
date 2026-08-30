import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import ScorePage from './pages/ScorePage';
import ModelsPage from './pages/ModelsPage';
import EvalPage from './pages/EvalPage';
import Disclaimer from './components/Disclaimer';

const NAV_ITEMS = [
  { to: '/', label: 'Score' },
  { to: '/models', label: 'Models' },
  { to: '/evaluation', label: 'Evaluation' },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-graphite-950">
        {/* Nav */}
        <nav className="border-b border-graphite-600 bg-graphite-800 px-6 py-3 flex items-center gap-8">
          <span className="font-display text-lg font-semibold text-paper-100 tracking-tight">
            HFG
          </span>
          <div className="flex gap-4">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `text-sm transition-colors ${
                    isActive
                      ? 'text-signal-teal'
                      : 'text-paper-400 hover:text-paper-100'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Content */}
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<ScorePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/evaluation" element={<EvalPage />} />
          </Routes>
        </main>

        {/* Persistent disclaimer */}
        <Disclaimer />
      </div>
    </BrowserRouter>
  );
}
