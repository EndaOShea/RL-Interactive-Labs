import React from 'react';

// Matrix heatmap (SVG). Modes: 'gray' (images/feature maps), 'heat' (0..1 dark→
// accent→white), 'diverging' (− red / + teal). Optional per-cell values + axis
// labels. Reused for conv kernels, attention, spectrograms, loss surfaces, etc.
export interface HeatmapProps {
  matrix: number[][];
  mode?: 'gray' | 'heat' | 'diverging';
  min?: number;
  max?: number;
  cell?: number;
  gap?: number;
  showValues?: boolean;
  rowLabels?: string[];
  colLabels?: string[];
  accent?: string;
}

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function rgb(r: number, g: number, b: number) { return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`; }

const Heatmap: React.FC<HeatmapProps> = ({ matrix, mode = 'heat', min, max, cell = 26, gap = 2, showValues, rowLabels, colLabels, accent = '#a855f7' }) => {
  const rows = matrix.length, cols = matrix[0]?.length ?? 0;
  let lo = min ?? Infinity, hi = max ?? -Infinity;
  if (min == null || max == null) matrix.forEach((r) => r.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  if (!isFinite(lo)) lo = 0; if (!isFinite(hi)) hi = 1;
  const absMax = Math.max(Math.abs(lo), Math.abs(hi)) || 1;

  const color = (v: number) => {
    if (mode === 'diverging') {
      const t = Math.max(-1, Math.min(1, v / absMax));
      if (t >= 0) return rgb(lerp(20, 45, 1 - t), lerp(26, 212, t) + (1 - t) * 0, lerp(40, 191, t));
      const a = -t; return rgb(lerp(20, 248, a), lerp(26, 113, a), lerp(40, 113, a));
    }
    const t = hi > lo ? (v - lo) / (hi - lo) : 0;
    if (mode === 'gray') return rgb(lerp(12, 240, t), lerp(15, 244, t), lerp(22, 250, t));
    // heat: dark → accent → white
    const ar = 168, ag = 85, ab = 247;
    if (t < 0.5) { const u = t / 0.5; return rgb(lerp(12, ar, u), lerp(15, ag, u), lerp(22, ab, u)); }
    const u = (t - 0.5) / 0.5; return rgb(lerp(ar, 255, u), lerp(ag, 255, u), lerp(ab, 255, u));
  };

  const padL = rowLabels ? 30 : 4, padT = colLabels ? 18 : 4;
  const W = padL + cols * (cell + gap) + 4, H = padT + rows * (cell + gap) + 4;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {colLabels?.map((l, c) => <text key={c} x={padL + c * (cell + gap) + cell / 2} y={12} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill="var(--t2)">{l}</text>)}
      {rowLabels?.map((l, r) => <text key={r} x={padL - 6} y={padT + r * (cell + gap) + cell / 2 + 3} textAnchor="end" fontSize="9" fontFamily="var(--mono)" fill="var(--t2)">{l}</text>)}
      {matrix.map((row, r) => row.map((v, c) => {
        const x = padL + c * (cell + gap), y = padT + r * (cell + gap);
        const t = hi > lo ? (v - lo) / (hi - lo) : 0;
        return (
          <g key={`${r}-${c}`}>
            <rect x={x} y={y} width={cell} height={cell} rx={3} fill={color(v)} stroke="rgba(120,130,170,.12)" strokeWidth={0.6} />
            {showValues && cell >= 20 && <text x={x + cell / 2} y={y + cell / 2 + 3} textAnchor="middle" fontSize={Math.min(9.5, cell / 2.6)} fontFamily="var(--mono)" fill={t > 0.55 ? 'rgba(8,11,20,.85)' : 'var(--t1)'}>{Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}</text>}
          </g>
        );
      }))}
    </svg>
  );
};

export default Heatmap;
