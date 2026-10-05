/**
 * Persistent research-demo disclaimer (spec Section 5.13, view 4).
 * Always visible at the bottom of the canvas with subtle, clean enterprise styling.
 */
export default function Disclaimer() {
  return (
    <footer className="border-t border-slate-200 bg-white/80 py-4 px-6 text-center text-xs text-slate-500">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
        <p className="text-[11px] text-slate-500">
          <strong className="text-slate-800 font-semibold">Research Framework:</strong>{' '}
          Model-calibrated neuro-symbolic hallucination detection prototype. Calibrated verdicts reflect statistical pattern matching.
        </p>
        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
          <span>Stage 1: Features</span>
          <span>•</span>
          <span>Stage 2: Symbolic Gate</span>
          <span>•</span>
          <span>Stage 3: Escalation</span>
        </div>
      </div>
    </footer>
  );
}
