import React from 'react';
import { useTheme } from '../../utils/theme';

// Labelled 2-D map for PCA projections of word / document vectors (NLP labs).
// Equal scaling on both axes so the projected geometry is not stretched. The
// coordinates are PCA scores — only relative positions matter, so no ticks.

export interface MapPoint { x: number; y: number; label: string; color: string; faint?: boolean; bold?: boolean; }
export interface MapArrow { x1: number; y1: number; x2: number; y2: number; color: string; dash?: boolean; width?: number; head?: boolean; }
export interface MapRing { x: number; y: number; color: string; text?: string; }
export interface MapStar { x: number; y: number; color: string; label: string; }

export interface WordMapProps {
  points: MapPoint[];
  arrows?: MapArrow[];
  rings?: MapRing[];
  star?: MapStar | null;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}

const WordMap: React.FC<WordMapProps> = ({ points, arrows = [], rings = [], star, width = 520, height = 420, xLabel, yLabel }) => {
  const isLight = useTheme() === 'light';
  const padL = 30, padR = 16, padT = 14, padB = 30;
  const plotW = width - padL - padR, plotH = height - padT - padB;

  const xs = [...points.map((p) => p.x), ...(star ? [star.x] : []), ...arrows.flatMap((a) => [a.x1, a.x2])].filter(Number.isFinite);
  const ys = [...points.map((p) => p.y), ...(star ? [star.y] : []), ...arrows.flatMap((a) => [a.y1, a.y2])].filter(Number.isFinite);
  const x0 = Math.min(0, ...xs), x1 = Math.max(0, ...xs), y0 = Math.min(0, ...ys), y1 = Math.max(0, ...ys);
  const spanX = Math.max(1e-6, x1 - x0) * 1.24, spanY = Math.max(1e-6, y1 - y0) * 1.2;
  const scale = Math.min(plotW / spanX, plotH / spanY);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const sx = (x: number) => padL + plotW / 2 + (x - cx) * scale;
  const sy = (y: number) => padT + plotH / 2 - (y - cy) * scale;

  const head = (a: MapArrow) => {
    const X1 = sx(a.x1), Y1 = sy(a.y1), X2 = sx(a.x2), Y2 = sy(a.y2);
    const L = Math.hypot(X2 - X1, Y2 - Y1);
    if (L < 1e-6) return '';
    const ux = (X2 - X1) / L, uy = (Y2 - Y1) / L, s = 8;
    return `M ${X2} ${Y2} L ${X2 - s * ux + s * 0.45 * uy} ${Y2 - s * uy - s * 0.45 * ux} L ${X2 - s * ux - s * 0.45 * uy} ${Y2 - s * uy + s * 0.45 * ux} Z`;
  };
  const stroke = isLight ? 'rgba(255,255,255,.9)' : 'rgba(8,11,20,.75)';

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--border)" />
      {/* origin cross-hair = the PCA mean */}
      <line x1={sx(0)} y1={padT} x2={sx(0)} y2={padT + plotH} stroke={isLight ? 'rgba(50,60,90,.14)' : 'rgba(120,130,170,.12)'} strokeDasharray="3 4" />
      <line x1={padL} y1={sy(0)} x2={padL + plotW} y2={sy(0)} stroke={isLight ? 'rgba(50,60,90,.14)' : 'rgba(120,130,170,.12)'} strokeDasharray="3 4" />

      {arrows.map((a, i) => (
        <g key={`a${i}`}>
          <line x1={sx(a.x1)} y1={sy(a.y1)} x2={sx(a.x2)} y2={sy(a.y2)} stroke={a.color} strokeWidth={a.width ?? 2} strokeDasharray={a.dash ? '5 4' : undefined} strokeLinecap="round" opacity={0.9} />
          {a.head !== false && <path d={head(a)} fill={a.color} opacity={0.9} />}
        </g>
      ))}

      {rings.map((r, i) => (
        <g key={`r${i}`}>
          <circle cx={sx(r.x)} cy={sy(r.y)} r={9} fill="none" stroke={r.color} strokeWidth={1.8} />
          {r.text && <text x={sx(r.x) - 11} y={sy(r.y) - 8} textAnchor="end" fontSize="9.5" fontFamily="var(--mono)" fontWeight={700} fill={r.color}>{r.text}</text>}
        </g>
      ))}

      {points.map((p, i) => (
        <g key={`p${i}`} opacity={p.faint ? 0.38 : 1}>
          <circle cx={sx(p.x)} cy={sy(p.y)} r={p.bold ? 5.5 : 4.2} fill={p.color} stroke={stroke} strokeWidth={0.9} />
          <text x={sx(p.x) + 7} y={sy(p.y) + 3.5} fontSize={p.bold ? 11 : 10} fontFamily="var(--mono)" fontWeight={p.bold ? 700 : 400} fill={p.bold ? 'var(--t0)' : 'var(--t1)'}>{p.label}</text>
        </g>
      ))}

      {star && (
        <g>
          <path d={`M ${sx(star.x)} ${sy(star.y) - 8} L ${sx(star.x) + 2.4} ${sy(star.y) - 2.4} L ${sx(star.x) + 8} ${sy(star.y)} L ${sx(star.x) + 2.4} ${sy(star.y) + 2.4} L ${sx(star.x)} ${sy(star.y) + 8} L ${sx(star.x) - 2.4} ${sy(star.y) + 2.4} L ${sx(star.x) - 8} ${sy(star.y)} L ${sx(star.x) - 2.4} ${sy(star.y) - 2.4} Z`}
            fill={star.color} stroke={stroke} strokeWidth={0.8} />
          <text x={sx(star.x) + 10} y={sy(star.y) - 8} fontSize="10.5" fontFamily="var(--mono)" fontWeight={700} fill={star.color}>{star.label}</text>
        </g>
      )}

      {xLabel && <text x={padL + plotW / 2} y={height - 9} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)">{xLabel}</text>}
      {yLabel && <text x={12} y={padT + plotH / 2} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90 12 ${padT + plotH / 2})`}>{yLabel}</text>}
    </svg>
  );
};

export default WordMap;
