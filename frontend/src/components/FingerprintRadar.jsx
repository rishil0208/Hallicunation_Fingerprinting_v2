/**
 * FingerprintRadar — radial visualization adapted for modern light enterprise canvas.
 * Plots the 6 mathematical features (H, S, C, E, D, M) against model baseline centroids.
 */
import React from 'react';

const FEATURES = ['H', 'S', 'C', 'E', 'D', 'M'];
const LABELS = {
  H: 'Hedge Density',
  S: 'Specificity',
  C: 'Citation Vagueness',
  E: 'Evidence Density',
  D: 'Semantic Drift',
  M: 'Confidence Marker',
};

const VERDICT_COLORS = {
  LOW_RISK: '#059669',          // Emerald
  AMBIGUOUS: '#D97706',         // Amber
  HIGH_RISK: '#DC2626',         // Crimson
  RESOLVED_AMBIGUOUS: '#4F46E5',// Indigo
};

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  };
}

function buildPolygonPoints(cx, cy, values, maxR) {
  const step = 360 / values.length;
  return values
    .map((v, i) => {
      const { x, y } = polarToCartesian(cx, cy, v * maxR, i * step);
      return `${x},${y}`;
    })
    .join(' ');
}

export default function FingerprintRadar({
  featureBreakdown,
  fingerprint,
  verdict,
  size = 320,
}) {
  const cx = size / 2;
  const cy = size / 2;
  const maxR = size * 0.36;
  const step = 360 / FEATURES.length;

  // Calibrated normal region (from fingerprint centroid data)
  const calibratedValues = FEATURES.map((f) => {
    if (!fingerprint?.clusters?.centroids?.[0]) return 0.5;
    return fingerprint.clusters.centroids[0][f] ?? 0.5;
  });

  // Current answer values
  const currentValues = FEATURES.map((f) =>
    featureBreakdown ? Math.min(Math.max(featureBreakdown[f] ?? 0, 0), 1) : 0
  );

  const verdictColor = VERDICT_COLORS[verdict] || '#2563EB';

  // Ridge lines (layered concentric paths at low opacity)
  const ridgeLines = [0.35, 0.6, 0.85].map((scale) => {
    const vals = calibratedValues.map((v) => v * scale);
    return buildPolygonPoints(cx, cy, vals, maxR);
  });

  return (
    <div className="flex flex-col items-center">
      <div className="relative p-2 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="overflow-visible"
        >
          {/* Subtle Concentric Guide Rings */}
          {[0.25, 0.5, 0.75, 1.0].map((ring) => (
            <polygon
              key={ring}
              points={buildPolygonPoints(
                cx, cy,
                FEATURES.map(() => ring),
                maxR
              )}
              fill={ring === 1.0 ? '#F8FAFC' : 'none'}
              stroke="#E2E8F0"
              strokeWidth={ring === 1.0 ? '1.5' : '1'}
              strokeDasharray={ring < 1.0 ? '2 2' : 'none'}
            />
          ))}

          {/* Radial Axis Lines & Feature Labels */}
          {FEATURES.map((f, i) => {
            const angle = i * step;
            const end = polarToCartesian(cx, cy, maxR, angle);
            const label = polarToCartesian(cx, cy, maxR + 22, angle);
            return (
              <g key={f}>
                <line
                  x1={cx} y1={cy}
                  x2={end.x} y2={end.y}
                  stroke="#CBD5E1"
                  strokeWidth="1"
                />
                <circle cx={label.x} cy={label.y} r="12" fill="#F1F5F9" stroke="#E2E8F0" strokeWidth="1" />
                <text
                  x={label.x} y={label.y}
                  fill="#1E293B"
                  fontSize="10"
                  fontWeight="700"
                  fontFamily="'JetBrains Mono', monospace"
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {f}
                </text>
              </g>
            );
          })}

          {/* Calibrated Model Baseline (Shaded Region) */}
          {ridgeLines.map((points, i) => (
            <polygon
              key={i}
              points={points}
              fill="#3B82F6"
              fillOpacity={0.05 + i * 0.04}
              stroke="#3B82F6"
              strokeWidth="1"
              strokeOpacity={0.3}
            />
          ))}

          {/* Current Answer Overlay Polygon */}
          {featureBreakdown && (
            <polygon
              points={buildPolygonPoints(cx, cy, currentValues, maxR)}
              fill={verdictColor}
              fillOpacity="0.18"
              stroke={verdictColor}
              strokeWidth="2.5"
              strokeLinejoin="round"
              className="radar-line"
            />
          )}

          {/* Feature Value Vertex Dots */}
          {featureBreakdown &&
            currentValues.map((v, i) => {
              const { x, y } = polarToCartesian(cx, cy, v * maxR, i * step);
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r="5" fill="#FFFFFF" stroke={verdictColor} strokeWidth="2.5" />
                  <circle cx={x} cy={y} r="2" fill={verdictColor} />
                </g>
              );
            })}
        </svg>
      </div>

      {/* Feature Breakdown Metrics Grid */}
      {featureBreakdown && (
        <div className="mt-4 w-full grid grid-cols-2 sm:grid-cols-3 gap-2">
          {FEATURES.map((f) => {
            const val = featureBreakdown[f] ?? 0;
            return (
              <div
                key={f}
                className="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-200/80 text-xs"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-5 h-5 flex items-center justify-center rounded font-mono font-bold bg-white text-slate-700 border border-slate-200 text-[10px]">
                    {f}
                  </span>
                  <span className="text-slate-600 text-[11px] truncate">
                    {LABELS[f]}
                  </span>
                </div>
                <span
                  className="font-mono font-semibold ml-2 text-xs"
                  style={{ color: verdictColor }}
                >
                  {val.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
