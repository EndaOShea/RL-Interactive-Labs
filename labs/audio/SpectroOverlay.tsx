import React from 'react';
import { useTheme } from '../../utils/theme';

// Area-local overlay for the spectrogram lab: draws the per-frame peak-frequency
// trace (the dominant bin over time) as a poly-line on top of the Heatmap, plus
// a moving playhead at the current column. Purely additive — it sits absolutely
// over the shared Heatmap and never modifies it.
//
// rows are displayed HIGH frequency at the top (row 0) → LOW at the bottom, so a
// display row r corresponds to frequency bin (nBins-1-r). The trace y is in
// display-row space.

export interface SpectroOverlayProps {
  peakRows: number[];   // display-row index of the peak bin, per filled column (-1 = empty)
  nCols: number;
  nRows: number;
  cell: number;
  gap: number;
  current: number;      // current column (playhead)
  color?: string;
}

const SpectroOverlay: React.FC<SpectroOverlayProps> = ({ peakRows, nCols, nRows, cell, gap, current, color = '#fde68a' }) => {
  const isLight = useTheme() === 'light';
  const w = nCols * (cell + gap);
  const h = nRows * (cell + gap);
  const cx = (c: number) => c * (cell + gap) + cell / 2;
  const cy = (r: number) => r * (cell + gap) + cell / 2;

  const pts = peakRows
    .map((r, c) => (r >= 0 ? `${cx(c)},${cy(r)}` : null))
    .filter((p): p is string => p !== null)
    .join(' ');

  const playX = current > 0 ? (current - 1) * (cell + gap) + cell / 2 : -10;

  return (
    <svg
      width={w} height={h} viewBox={`0 0 ${w} ${h}`}
      style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', overflow: 'visible' }}
    >
      {pts && <polyline points={pts} fill="none" stroke={color} strokeWidth={1.6} strokeOpacity={0.85} strokeLinejoin="round" strokeLinecap="round" />}
      {peakRows.map((r, c) => (r >= 0 ? (
        <circle key={c} cx={cx(c)} cy={cy(r)} r={1.7} fill={color} fillOpacity={0.9} />
      ) : null))}
      {current > 0 && current <= nCols && (
        <line x1={playX} y1={0} x2={playX} y2={h} stroke={isLight ? 'var(--t0)' : '#fff'} strokeOpacity={0.35} strokeWidth={1} strokeDasharray="3 3" />
      )}
    </svg>
  );
};

export default SpectroOverlay;
