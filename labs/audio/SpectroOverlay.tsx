import React from 'react';

// Area-local overlay for the spectrogram lab, drawn INSIDE the spectrogram's own
// SVG (so it shares its viewBox and scales with it): the per-frame peak trace
// (the loudest bin/band of each computed column) and an outline around the
// column the math panel is describing. Coordinates are already in SVG pixels,
// mapped from physical time (s) and frequency (Hz) by the lab.

export interface TracePoint { x: number; y: number; }

export interface SpectroOverlayProps {
  trace: (TracePoint | null)[];                             // one entry per computed column (null = no peak, e.g. a silent frame)
  frame: { x0: number; x1: number; y0: number; y1: number } | null; // current column
  traceColor: string;
  frameColor: string;
}

const SpectroOverlay: React.FC<SpectroOverlayProps> = ({ trace, frame, traceColor, frameColor }) => {
  // Split the trace at gaps so a silent frame is not bridged by a line.
  const runs: TracePoint[][] = [];
  let cur: TracePoint[] = [];
  trace.forEach((p) => {
    if (p) cur.push(p);
    else if (cur.length) { runs.push(cur); cur = []; }
  });
  if (cur.length) runs.push(cur);

  return (
    <g pointerEvents="none">
      {runs.map((run, i) => (
        <polyline key={i} points={run.map((p) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={traceColor} strokeWidth={1.6} strokeOpacity={0.9} strokeLinejoin="round" strokeLinecap="round" />
      ))}
      {trace.map((p, i) => (p ? <circle key={i} cx={p.x} cy={p.y} r={1.8} fill={traceColor} /> : null))}
      {frame && (
        <rect x={frame.x0} y={frame.y0} width={Math.max(1, frame.x1 - frame.x0)} height={frame.y1 - frame.y0} fill="none" stroke={frameColor} strokeWidth={1.6} />
      )}
    </g>
  );
};

export default SpectroOverlay;
