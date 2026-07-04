import React from 'react';
import { CLASS_COLORS } from '../../components/labkit/viz/ScatterPlot';
import { useTheme } from '../../utils/theme';

// Area-local OPTICS reachability plot: a bar per point in the reachability
// ordering, height = reachability distance. Valleys = clusters; tall bars are
// the cluster boundaries / noise. Mirrors the Cinematic-Stage palette and is
// purely presentational (the DBSCAN/OPTICS lab owns the ordering + labels).

export interface ReachabilityPlotProps {
  /** Reachability distance per ordered point (Infinity → undefined → drawn full-height/noise). */
  reach: number[];
  /** Cluster label per ordered point (-1 = noise). */
  labels: number[];
  /** How many ordered points have been revealed so far (for the animated sweep). */
  revealed?: number;
  /** Extraction threshold ξ drawn as a horizontal line (data units = same as reach). */
  threshold?: number;
  width?: number;
  height?: number;
  accent?: string;
}

const ReachabilityPlot: React.FC<ReachabilityPlotProps> = ({
  reach, labels, revealed, threshold, width = 440, height = 150, accent = '#f472b6',
}) => {
  const isLight = useTheme() === 'light';
  const padL = 30, padR = 10, padT = 12, padB = 18;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const n = Math.max(1, reach.length);
  const finite = reach.filter((r) => Number.isFinite(r));
  const maxR = Math.max(1e-6, ...finite, threshold ?? 0);
  const bw = plotW / n;
  const rev = revealed == null ? n : revealed;
  const sy = (r: number) => plotT(r);
  function plotT(r: number) {
    const clamped = Number.isFinite(r) ? Math.min(r, maxR) : maxR;
    return (clamped / maxR) * plotH;
  }
  const thY = threshold != null ? padT + plotH - plotT(threshold) : null;

  return (
    <svg
      width={width} height={height} viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}
    >
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--border)" strokeWidth="1" />
      {[0.5].map((t) => (
        <line key={t} x1={padL} y1={padT + t * plotH} x2={padL + plotW} y2={padT + t * plotH} stroke={isLight ? 'rgba(50,60,90,.08)' : 'rgba(120,130,170,.08)'} />
      ))}

      {reach.map((r, i) => {
        const h = sy(r);
        const isRev = i < rev;
        const lab = labels[i];
        const col = lab >= 0 ? CLASS_COLORS[lab % CLASS_COLORS.length] : 'var(--t2)';
        return (
          <rect
            key={i}
            x={padL + i * bw} y={padT + plotH - h}
            width={Math.max(0.6, bw - 0.5)} height={h}
            fill={col} opacity={isRev ? 0.9 : 0.18}
          />
        );
      })}

      {thY != null && (
        <line x1={padL} y1={thY} x2={padL + plotW} y2={thY} stroke={accent} strokeWidth={1.4} strokeDasharray="5 4" opacity={0.85} />
      )}

      <text x={padL + plotW / 2} y={height - 5} textAnchor="middle" fill="var(--t2)" fontSize="9.5" fontFamily="var(--mono)">reachability ordering →</text>
      <text x={11} y={padT + plotH / 2} textAnchor="middle" fill="var(--t2)" fontSize="9.5" fontFamily="var(--mono)" transform={`rotate(-90 11 ${padT + plotH / 2})`}>reach-dist</text>
    </svg>
  );
};

export default ReachabilityPlot;
