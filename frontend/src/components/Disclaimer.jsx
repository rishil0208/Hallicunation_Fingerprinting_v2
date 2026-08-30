/**
 * Persistent research-demo disclaimer (spec Section 5.13, view 4).
 * Not a dismissible modal — always visible.
 */
export default function Disclaimer() {
  return (
    <div className="border-t border-graphite-600 bg-graphite-800 px-6 py-3 text-xs text-paper-400">
      <strong className="text-paper-100">Research Prototype.</strong>{' '}
      This is a research demonstration of per-model hallucination fingerprinting,
      not a production fact-checking system. Verdicts reflect statistical pattern
      analysis on calibration data and should not be treated as ground truth.
      See Section 5.11 of the project specification.
    </div>
  );
}
