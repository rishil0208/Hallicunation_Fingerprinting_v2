/**
 * FingerprintRadar — the signature visual element (spec Section 5.13).
 *
 * A radial/polar plot with six axes (H, S, C, E, D, M) that visually
 * resembles a fingerprint ridge pattern. The model's calibrated normal
 * region is drawn as a filled teal ring; the current answer's feature
 * vector is overlaid as a sharp line colored by verdict state.
 */
import { useEffect, useRef } from 'react';

const FEATURES = ['H', 'S', 'C', 'E', 'D', 'M'];
const LABELS = {
  H: 'Hedge',
  S: 'Specificity',
  C: 'Citation',
  E: 'Evidence',
  D: 'Drift',
  M: 'Confidence',
};

const VERDICT_COLORS = {
  LOW_RISK: '#4FE3C1',
  AMBIGUOUS: '#F2B84B',
  HIGH_RISK: '#FF6B4A',
  RESOLVED_AMBIGUOUS: '#F2B84B',
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
  const maxR = size * 0.38;
  const step = 360 / FEATURES.length;

  // Calibrated normal region (from fingerprint centroid data)
  const calibratedValues = FEATURES.map((f) => {
    if (!fingerprint?.clusters?.centroids?.[0]) return 0.5;
    return fingerprint.clusters.centroids[0][f] ?? 0.5;
  });

  // Current answer values
  const currentValues = FEATURES.map((f) =>
    featureBreakdown ? Math.min(featureBreakdown[f] ?? 0, 1) : 0
  );

  const verdictColor = VERDICT_COLORS[verdict] || '#8B93A1';

  // Ridge lines (layered concentric paths at low opacity)
  const ridgeLines = [0.3, 0.5, 0.7, 0.9].map((scale) => {
    const vals = calibratedValues.map((v) => v * scale);
    return buildPolygonPoints(cx, cy, vals, maxR);
  });

  return (
    <div className="flex flex-col items-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="drop-shadow-lg"
      >
        {/* Grid lines */}
        {[0.25, 0.5, 0.75, 1.0].map((ring) => (
          <polygon
            key={ring}
            points={buildPolygonPoints(
              cx, cy,
              FEATURES.map(() => ring),
              maxR
            )}
            fill="none"
            stroke="#262B33"
            strokeWidth="0.5"
          />
        ))}

        {/* Axis lines + labels */}
        {FEATURES.map((f, i) => {
          const angle = i * step;
          const end = polarToCartesian(cx, cy, maxR, angle);
          const label = polarToCartesian(cx, cy, maxR + 20, angle);
          return (
            <g key={f}>
              <line
                x1={cx} y1={cy}
                x2={end.x} y2={end.y}
                stroke="#262B33"
                strokeWidth="0.5"
              />
              <text
                x={label.x} y={label.y}
                fill="#8B93A1"
                fontSize="11"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {f}
              </text>
            </g>
          );
        })}

        {/* Calibrated region (ridge lines) */}
        {ridgeLines.map((points, i) => (
          <polygon
            key={i}
            points={points}
            fill="#4FE3C1"
            fillOpacity={0.06 + i * 0.03}
            stroke="#4FE3C1"
            strokeWidth="0.5"
            strokeOpacity={0.2}
          />
        ))}

        {/* Current answer overlay */}
        {featureBreakdown && (
          <polygon
            points={buildPolygonPoints(cx, cy, currentValues, maxR)}
            fill="none"
            stroke={verdictColor}
            strokeWidth="2"
            strokeLinejoin="round"
            className="radar-line"
          />
        )}

        {/* Feature value dots */}
        {featureBreakdown &&
          currentValues.map((v, i) => {
            const { x, y } = polarToCartesian(cx, cy, v * maxR, i * step);
            return (
              <circle
                key={i}
                cx={x} cy={y} r="3"
                fill={verdictColor}
              />
            );
          })}
      </svg>

      {/* Feature legend */}
      {featureBreakdown && (
        <div className="mt-3 grid grid-cols-3 gap-x-6 gap-y-1 text-xs">
          {FEATURES.map((f) => (
            <div key={f} className="flex items-center gap-1.5">
              <span className="font-mono text-paper-400">{f}</span>
              <span className="text-paper-100">
                {LABELS[f]}:{' '}
                <span className="font-mono" style={{ color: verdictColor }}>
                  {(featureBreakdown[f] ?? 0).toFixed(3)}
                </span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
