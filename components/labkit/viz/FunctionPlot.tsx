import React from 'react';
import { ACC } from '../../stage/primitives';
import { useTheme } from '../../../utils/theme';

// Generic function / line plot (SVG). Powers linear-regression (data + fitted
// line, loss curve) and future activation / optimizer / calculus labs.

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

const FunctionPlot: React.FC<FunctionPlotProps> = ({
  series = [], scatter = [], markers = [], domain = [0, 1], range = [0, 1],
  width = 520, height = 460, showAxes = true, xLabel, yLabel,
}) => {
  const isLight = useTheme() === 'light';
  const padL = 44, padR = 14, padT = 14, padB = 36;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const [dx0, dx1] = domain;
  const [dy0, dy1] = range;
  const sx = (x: number) => padL + ((x - dx0) / (dx1 - dx0)) * plotW;
  const sy = (y: number) => padT + (1 - (y - dy0) / (dy1 - dy0)) * plotH;
  const fmtTick = (v: number) => (Math.abs(v) < 1e-9 ? '0' : parseFloat(v.toFixed(2)).toString());
  const clampY = (y: number) => Math.max(padT, Math.min(padT + plotH, y));

  const toPath = (pts: PlotPoint[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x)} ${clampY(sy(p.y))}`).join(' ');

  return (
    <svg
      width={width} height={height} viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block', borderRadius: 14, background: 'var(--bg2)', border: '1px solid var(--border)', maxWidth: '100%' }}
    >
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

      {/* series (area fill then stroke) */}
      {series.map((s, i) => {
        const color = s.color || ACC;
        const d = toPath(s.points);
        return (
          <g key={i}>
            {s.area && s.points.length > 1 && (
              <path
                d={`${d} L ${sx(s.points[s.points.length - 1].x)} ${padT + plotH} L ${sx(s.points[0].x)} ${padT + plotH} Z`}
                fill={color} opacity={0.1}
              />
            )}
            <path d={d} fill="none" stroke={color} strokeWidth={s.width ?? 2.2} strokeDasharray={s.dash ? '5 5' : undefined} strokeLinejoin="round" strokeLinecap="round" />
          </g>
        );
      })}

      {/* scatter */}
      {scatter.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={clampY(sy(p.y))} r={p.r ?? 4} fill={p.color || 'var(--t1)'} opacity={0.9} stroke={isLight ? 'rgba(255,255,255,.85)' : 'rgba(8,11,20,.7)'} strokeWidth="0.8" />
      ))}

      {/* markers */}
      {markers.map((m, i) => (
        <g key={i}>
          <circle cx={sx(m.x)} cy={clampY(sy(m.y))} r={m.r ?? 5} fill={isLight ? 'var(--t0)' : '#fff'} stroke={m.color || ACC} strokeWidth="2" />
          {m.label && <text x={sx(m.x) + 8} y={clampY(sy(m.y)) - 6} fill="var(--t1)" fontSize="10" fontFamily="var(--mono)">{m.label}</text>}
        </g>
      ))}

      {xLabel && <text x={padL + plotW / 2} y={height - 8} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)">{xLabel}</text>}
      {yLabel && <text x={12} y={padT + plotH / 2} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90 12 ${padT + plotH / 2})`}>{yLabel}</text>}
    </svg>
  );
};

export default FunctionPlot;
