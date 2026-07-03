import React from 'react';
import { ACC } from '../../stage/primitives';
import { useTheme } from '../../../utils/theme';

// Horizontal probability/value bars with labels. Used for LLM next-token
// distributions, softmax, attention rows, class probabilities, spectra.
export interface Bar { label: string; value: number; color?: string; highlight?: boolean; muted?: boolean; }
export interface DistributionBarsProps {
  bars: Bar[];
  width?: number;
  rowH?: number;
  max?: number;
  accent?: string;
  valueFmt?: (v: number) => string;
}

const DistributionBars: React.FC<DistributionBarsProps> = ({ bars, width = 360, rowH = 26, max, accent = ACC, valueFmt }) => {
  const isLight = useTheme() === 'light';
  const hi = max ?? Math.max(1e-9, ...bars.map((b) => b.value));
  const labelW = 64, valW = 52, gap = 8;
  const barMax = width - labelW - valW - gap * 2;
  const fmt = valueFmt ?? ((v: number) => v.toFixed(3));
  const H = bars.length * rowH + 6;

  return (
    <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {bars.map((b, i) => {
        const y = i * rowH + 3;
        const w = Math.max(0, (b.value / hi) * barMax);
        const col = b.color || accent;
        const op = b.muted ? 0.3 : 1;
        return (
          <g key={i} opacity={op}>
            <text x={labelW - 8} y={y + rowH / 2 + 3} textAnchor="end" fontSize="12" fontFamily="var(--mono)" fill={b.highlight ? (isLight ? 'var(--t0)' : '#fff') : 'var(--t1)'} fontWeight={b.highlight ? 700 : 400}>{b.label}</text>
            <rect x={labelW} y={y + 3} width={barMax} height={rowH - 9} rx={4} fill="var(--bg2)" />
            <rect x={labelW} y={y + 3} width={w} height={rowH - 9} rx={4} fill={col} stroke={b.highlight ? (isLight ? 'var(--t0)' : '#fff') : 'none'} strokeWidth={b.highlight ? 1.4 : 0} />
            <text x={labelW + barMax + gap} y={y + rowH / 2 + 3} fontSize="11" fontFamily="var(--mono)" fill="var(--t2)">{fmt(b.value)}</text>
          </g>
        );
      })}
    </svg>
  );
};

export default DistributionBars;
