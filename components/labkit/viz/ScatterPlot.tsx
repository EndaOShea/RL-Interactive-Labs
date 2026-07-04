import React, { useMemo, useRef } from 'react';
import { ACC, GOOD } from '../../stage/primitives';
import { useTheme } from '../../../utils/theme';

// Generic 2-D scatter plot (SVG). Powers k-NN, logistic regression, k-means and
// PCA. Pure/presentational: the lab owns the data + animation, this renders one
// frame. Reuses the Cinematic-Stage palette.

export interface ScatterPoint { x: number; y: number; cls?: number; size?: number; faint?: boolean; }
export interface ScatterLine { x1: number; y1: number; x2: number; y2: number; color?: string; dash?: boolean; width?: number; }
export interface ScatterMarker { x: number; y: number; cls?: number; color?: string; r?: number; ring?: boolean; }
export interface ScatterCircle { x: number; y: number; r: number; color?: string; } // r in data units
export interface ScatterEllipse { cx: number; cy: number; rx: number; ry: number; angle: number; color?: string; } // rx/ry data units, angle radians

export const CLASS_COLORS = [ACC, GOOD, '#fbbf24', '#f87171', '#38bdf8', '#fb7185'];

export interface ScatterPlotProps {
  points: ScatterPoint[];
  domain?: [number, number];           // x range (default [0,1])
  range?: [number, number];            // y range (default [0,1])
  width?: number;
  height?: number;
  classColors?: string[];
  /** Shaded decision field: class index (or -1 for none) at a data coordinate. */
  classify?: (x: number, y: number) => number;
  /** Memo key for the (expensive) decision field — bump when classify changes. */
  fieldKey?: string | number;
  fieldResolution?: number;            // grid cells per axis (default 26)
  centroids?: ScatterMarker[];
  markers?: ScatterMarker[];           // generic emphasised points (e.g. k-NN neighbours)
  lines?: ScatterLine[];
  circles?: ScatterCircle[];           // data-space circles (e.g. DBSCAN eps neighbourhood)
  ellipses?: ScatterEllipse[];         // data-space ellipses (e.g. Gaussian covariance)
  showAxes?: boolean;
  xLabel?: string;
  yLabel?: string;
  onAddPoint?: (x: number, y: number) => void;
}

const ScatterPlot: React.FC<ScatterPlotProps> = ({
  points, domain = [0, 1], range = [0, 1], width = 520, height = 460,
  classColors = CLASS_COLORS, classify, fieldKey, fieldResolution = 26,
  centroids, markers, lines, circles, ellipses, showAxes = true, xLabel, yLabel, onAddPoint,
}) => {
  const ref = useRef<SVGSVGElement | null>(null);
  const isLight = useTheme() === 'light';
  const padL = 44, padR = 14, padT = 14, padB = 36;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const [dx0, dx1] = domain;
  const [dy0, dy1] = range;

  const sx = (x: number) => padL + ((x - dx0) / (dx1 - dx0)) * plotW;
  const sy = (y: number) => padT + (1 - (y - dy0) / (dy1 - dy0)) * plotH;
  const fmtTick = (v: number) => (Math.abs(v) < 1e-9 ? '0' : parseFloat(v.toFixed(2)).toString());
  const colorOf = (cls?: number) => (cls == null || cls < 0 ? 'var(--t2)' : classColors[cls % classColors.length]);

  // Decision field (memoised — recompute only when fieldKey/geometry changes).
  const field = useMemo(() => {
    if (!classify) return null;
    const n = fieldResolution;
    const cw = plotW / n, ch = plotH / n;
    const cells: { x: number; y: number; w: number; h: number; cls: number }[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const cx = dx0 + ((i + 0.5) / n) * (dx1 - dx0);
        const cy = dy0 + ((j + 0.5) / n) * (dy1 - dy0);
        const cls = classify(cx, cy);
        if (cls < 0) continue;
        cells.push({ x: padL + i * cw, y: padT + (n - 1 - j) * ch, w: cw + 0.6, h: ch + 0.6, cls });
      }
    }
    return cells;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fieldKey, fieldResolution, width, height, dx0, dx1, dy0, dy1]);

  const handleClick = (e: React.MouseEvent) => {
    if (!onAddPoint || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const py = ((e.clientY - rect.top) / rect.height) * height;
    if (px < padL || px > width - padR || py < padT || py > height - padB) return;
    const x = dx0 + ((px - padL) / plotW) * (dx1 - dx0);
    const y = dy0 + (1 - (py - padT) / plotH) * (dy1 - dy0);
    onAddPoint(x, y);
  };

  return (
    <svg
      ref={ref}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      onClick={handleClick}
      style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', cursor: onAddPoint ? 'crosshair' : 'default', maxWidth: '100%' }}
    >
      {/* decision field */}
      {field?.map((c, i) => (
        <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={classColors[c.cls % classColors.length]} opacity={0.16} />
      ))}

      {/* plot frame + grid + numeric ticks */}
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

      {/* lines (boundaries, axes, links) */}
      {lines?.map((l, i) => (
        <line
          key={i} x1={sx(l.x1)} y1={sy(l.y1)} x2={sx(l.x2)} y2={sy(l.y2)}
          stroke={l.color || 'var(--t0)'} strokeWidth={l.width ?? 2}
          strokeDasharray={l.dash ? '5 5' : undefined} strokeLinecap="round"
        />
      ))}

      {/* data-space circles (e.g. DBSCAN eps neighbourhood) */}
      {circles?.map((c, i) => (
        <circle key={`c${i}`} cx={sx(c.x)} cy={sy(c.y)} r={c.r * plotW} fill="none" stroke={c.color || 'var(--t1)'} strokeWidth={1.2} strokeDasharray="4 4" opacity={0.75} />
      ))}

      {/* emphasis markers (e.g. k-NN neighbourhood rings) */}
      {markers?.map((m, i) => (
        <circle
          key={i} cx={sx(m.x)} cy={sy(m.y)} r={m.r ?? 11}
          fill={m.ring ? 'none' : (m.color || colorOf(m.cls))}
          stroke={m.color || colorOf(m.cls)} strokeWidth={m.ring ? 1.6 : 0}
          opacity={m.ring ? 0.9 : 0.5}
        />
      ))}

      {/* data points */}
      {points.map((p, i) => (
        <circle
          key={i} cx={sx(p.x)} cy={sy(p.y)} r={p.size ?? 4.5}
          fill={colorOf(p.cls)} opacity={p.faint ? 0.4 : 0.95}
          stroke={isLight ? 'rgba(255,255,255,.85)' : 'rgba(8,11,20,.7)'} strokeWidth="0.8"
        />
      ))}

      {/* data-space ellipses (e.g. Gaussian covariance) */}
      {ellipses?.map((e, i) => {
        const cx = sx(e.cx), cy = sy(e.cy);
        const deg = (-e.angle * 180) / Math.PI; // data-space y is up; SVG y is down
        const col = e.color || ACC;
        return (
          <ellipse key={`e${i}`} cx={cx} cy={cy} rx={e.rx * plotW} ry={e.ry * plotW} transform={`rotate(${deg} ${cx} ${cy})`} fill={col} fillOpacity={0.1} stroke={col} strokeWidth={1.8} />
        );
      })}

      {/* centroids */}
      {centroids?.map((c, i) => (
        <g key={i}>
          <circle cx={sx(c.x)} cy={sy(c.y)} r={(c.r ?? 9) + 4} fill="none" stroke={c.color || colorOf(c.cls)} strokeWidth="1.5" opacity={0.6} />
          <path
            d={`M ${sx(c.x) - 7} ${sy(c.y)} L ${sx(c.x) + 7} ${sy(c.y)} M ${sx(c.x)} ${sy(c.y) - 7} L ${sx(c.x)} ${sy(c.y) + 7}`}
            stroke={c.color || colorOf(c.cls)} strokeWidth="2.4" strokeLinecap="round"
          />
        </g>
      ))}

      {/* axis labels */}
      {xLabel && <text x={padL + plotW / 2} y={height - 8} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)">{xLabel}</text>}
      {yLabel && <text x={12} y={padT + plotH / 2} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90 12 ${padT + plotH / 2})`}>{yLabel}</text>}
    </svg>
  );
};

export default ScatterPlot;
