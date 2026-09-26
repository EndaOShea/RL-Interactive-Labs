import React, { useId } from 'react';
import { ACC } from '../../stage/primitives';
import { useTheme } from '../../../utils/theme';

// Generic function / line plot (SVG). Powers linear-regression (data + fitted
// line, loss curve) and future activation / optimizer / calculus labs.
//
// Geometry is exact: lines, areas and scatter are CLIPPED to the plot box (an
// SVG clipPath), never clamped vertex-by-vertex — clamping an endpoint of a
// straight line changes its slope, so a tangent that leaves the y-range would be
// drawn with the wrong gradient. Non-finite y values (NaN/±Infinity) break a
// series into separate segments instead of being joined through. A marker whose
// y is outside the range is drawn as a chevron pinned to the edge it left
// through, so it stays visible without claiming a false position.

export interface PlotPoint { x: number; y: number; }
export interface PlotSeries {
  points: PlotPoint[];
  color?: string;
  width?: number;
  dash?: boolean;
  area?: boolean;
}
export interface PlotScatter { x: number; y: number; color?: string; r?: number; }
export interface PlotMarker { x: number; y: number; color?: string; r?: number; label?: string; }

export interface FunctionPlotProps {
  series?: PlotSeries[];
  scatter?: PlotScatter[];
  markers?: PlotMarker[];
  domain?: [number, number];
  range?: [number, number];
  width?: number;
  height?: number;
  showAxes?: boolean;
  xLabel?: string;
  yLabel?: string;
}

// Pixel safety net against SVG overflow for astronomically large values. Far
// outside the visible box, so it cannot visibly alter any slope inside it.
const PX_LIMIT = 1e6;
const safePx = (v: number) => Math.max(-PX_LIMIT, Math.min(PX_LIMIT, v));

/** Split a series into contiguous runs of finite points. */
function segmentsOf(pts: PlotPoint[]): PlotPoint[][] {
  const segs: PlotPoint[][] = [];
  let cur: PlotPoint[] = [];
  for (const p of pts) {
    if (Number.isFinite(p.x) && Number.isFinite(p.y)) cur.push(p);
    else if (cur.length) { segs.push(cur); cur = []; }
  }
  if (cur.length) segs.push(cur);
  return segs;
}

const FunctionPlot: React.FC<FunctionPlotProps> = ({
  series = [], scatter = [], markers = [], domain = [0, 1], range = [0, 1],
  width = 520, height = 460, showAxes = true, xLabel, yLabel,
}) => {
  const isLight = useTheme() === 'light';
  const clipId = `fp-clip-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const padL = 44, padR = 14, padT = 14, padB = 36;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const [dx0, dx1] = domain;
  const [dy0, dy1] = range;
  const sx = (x: number) => safePx(padL + ((x - dx0) / (dx1 - dx0)) * plotW);
  const sy = (y: number) => safePx(padT + (1 - (y - dy0) / (dy1 - dy0)) * plotH);
  const fmtTick = (v: number) => (Math.abs(v) < 1e-9 ? '0' : parseFloat(v.toFixed(2)).toString());
  const yLo = Math.min(dy0, dy1), yHi = Math.max(dy0, dy1);
  const baseY = padT + plotH;

  const toPath = (pts: PlotPoint[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x)} ${sy(p.y)}`).join(' ');

  return (
    <svg
      width={width} height={height} viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={padL} y={padT} width={plotW} height={plotH} />
        </clipPath>
      </defs>
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--border)" strokeWidth="1" />
      {showAxes && [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const xpos = padL + t * plotW;
        const ypos = padT + (1 - t) * plotH;
        const xv = dx0 + t * (dx1 - dx0);
        const yv = dy0 + t * (dy1 - dy0);
        const interior = t > 0 && t < 1;
        return (
          <g key={t}>
            {interior && <line x1={xpos} y1={padT} x2={xpos} y2={padT + plotH} stroke={isLight ? 'rgba(50,60,90,.12)' : 'rgba(120,130,170,.08)'} />}
            {interior && <line x1={padL} y1={ypos} x2={padL + plotW} y2={ypos} stroke={isLight ? 'rgba(50,60,90,.12)' : 'rgba(120,130,170,.08)'} />}
            <line x1={xpos} y1={padT + plotH} x2={xpos} y2={padT + plotH + 3} stroke="var(--border)" />
            <text x={xpos} y={padT + plotH + 14} textAnchor="middle" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{fmtTick(xv)}</text>
            <line x1={padL - 3} y1={ypos} x2={padL} y2={ypos} stroke="var(--border)" />
            <text x={padL - 6} y={ypos + 3} textAnchor="end" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{fmtTick(yv)}</text>
          </g>
        );
      })}

      {/* series (area fill then stroke), clipped to the plot box */}
      <g clipPath={`url(#${clipId})`}>
        {series.map((s, i) => {
          const color = s.color || ACC;
          return (
            <g key={i}>
              {segmentsOf(s.points).map((seg, j) => {
                const d = toPath(seg);
                const first = seg[0], last = seg[seg.length - 1];
                return (
                  <g key={j}>
                    {s.area && seg.length > 1 && first && last && (
                      <path d={`${d} L ${sx(last.x)} ${baseY} L ${sx(first.x)} ${baseY} Z`} fill={color} opacity={0.1} />
                    )}
                    <path d={d} fill="none" stroke={color} strokeWidth={s.width ?? 2.2} strokeDasharray={s.dash ? '5 5' : undefined} strokeLinejoin="round" strokeLinecap="round" />
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* scatter (points outside the box are clipped, not pinned) */}
        {scatter.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={p.r ?? 4} fill={p.color || 'var(--t1)'} opacity={0.9} stroke={isLight ? 'rgba(255,255,255,.85)' : 'rgba(8,11,20,.7)'} strokeWidth="0.8" />
        ))}
      </g>

      {/* markers: exact position when in range; an edge chevron when the value is off-chart */}
      {markers.filter((m) => Number.isFinite(m.x)).map((m, i) => {
        const color = m.color || ACC;
        const cx = Math.max(padL, Math.min(padL + plotW, sx(m.x)));
        if (Number.isFinite(m.y) && m.y >= yLo && m.y <= yHi) {
          const cy = sy(m.y);
          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r={m.r ?? 5} fill={isLight ? 'var(--t0)' : '#fff'} stroke={color} strokeWidth="2" />
              {m.label && <text x={cx + 8} y={cy - 6} fill="var(--t1)" fontSize="10" fontFamily="var(--mono)">{m.label}</text>}
            </g>
          );
        }
        // Off-chart: the value is above the top (or NaN/+∞) → chevron at the top edge pointing up; below → bottom edge pointing down.
        const above = !(m.y < yLo) ? (dy1 >= dy0) : (dy1 < dy0);
        const ey = above ? padT + 7 : padT + plotH - 7;
        const tip = above ? ey - 6 : ey + 6;
        return (
          <g key={i}>
            <path d={`M ${cx - 6} ${ey} L ${cx} ${tip} L ${cx + 6} ${ey}`} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            {m.label && <text x={cx + 9} y={ey + (above ? 8 : -2)} fill="var(--t1)" fontSize="10" fontFamily="var(--mono)">{m.label} (off-chart)</text>}
          </g>
        );
      })}

      {xLabel && <text x={padL + plotW / 2} y={height - 8} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)">{xLabel}</text>}
      {yLabel && <text x={12} y={padT + plotH / 2} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90 12 ${padT + plotH / 2})`}>{yLabel}</text>}
    </svg>
  );
};

export default FunctionPlot;
