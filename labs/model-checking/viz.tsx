import React from 'react';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { useTheme } from '../../utils/theme';

// Area-local visual helpers for the Model-Checking labs. These DECORATE the
// shared GraphCanvas (the state-space graph) with a small schematic of the
// *current* concrete state, so the viewer sees what the highlighted node means
// — process lanes for mutual exclusion, two river banks for the puzzle. Kept
// area-local so the shared viz primitives stay untouched.

const PANEL = 'rgba(8,11,20,.62)';
const ACCENT = '#fb7185';

/* ---------------- Mutual exclusion: two process lanes ---------------- */

// Lane colours by location code: idle grey, entry steps blue/violet, critical rose.
const LANE_COLOR: Record<string, string> = { I: '#64748b', F: '#a78bfa', T: '#a78bfa', W: '#38bdf8', C: '#fb7185' };

const ProcessLane: React.FC<{ name: string; lanes: { code: string; title: string }[]; pos: number; x: number }> = ({ name, lanes, pos, x }) => {
  const isLight = useTheme() === 'light';
  const cellW = 78, cellH = 24, gap = 5, top = 0;
  return (
    <g transform={`translate(${x},0)`}>
      <text x={0} y={top - 8} fontSize={11} fontFamily="var(--mono)" fill="var(--t1)" fontWeight={600}>{name}</text>
      {lanes.map((ln, i) => {
        const y = top + i * (cellH + gap);
        const here = pos === i;
        const col = LANE_COLOR[ln.code] ?? '#38bdf8';
        return (
          <g key={i}>
            <rect x={0} y={y} width={cellW} height={cellH} rx={6}
              fill={here ? col : (isLight ? 'rgba(50,60,90,.10)' : 'rgba(120,130,170,.10)')}
              stroke={here ? col : 'var(--border)'} strokeWidth={here ? 1.6 : 1}
              style={here ? { filter: `drop-shadow(0 0 6px ${col})` } : undefined} />
            <text x={cellW / 2} y={y + cellH / 2 + 4} textAnchor="middle" fontSize={11}
              fontFamily="var(--disp)" fontWeight={600}
              fill={here ? 'rgba(8,11,20,.9)' : 'var(--t2)'}>{ln.title}</text>
          </g>
        );
      })}
    </g>
  );
};

/** Two process lanes for the highlighted state, plus the shared variables: the lock
 *  bit (lock protocol) or both flags and `turn` (Peterson variants). */
export const MutexSchematic: React.FC<{
  lanes: { code: string; title: string }[]; a: number; b: number; unsafe: boolean;
  lock?: boolean; peterson?: { flagA: boolean; flagB: boolean; turn: number };
}> = ({ lanes, a, b, unsafe, lock, peterson }) => {
  const isLight = useTheme() === 'light';
  const w = 240, h = 48 + lanes.length * 29;
  const vars = peterson
    ? `flag A=${peterson.flagA ? 1 : 0}  flag B=${peterson.flagB ? 1 : 0}  turn=${peterson.turn === 0 ? 'A' : 'B'}`
    : lock === undefined ? 'no shared lock' : lock ? '🔒 lock held' : 'lock free';
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <rect x={0} y={0} width={w} height={h} rx={12} fill={isLight ? 'var(--bg2)' : PANEL}
        stroke={unsafe ? (isLight ? 'var(--bad)' : '#f87171') : 'var(--border)'} strokeWidth={unsafe ? 1.8 : 1} />
      <g transform="translate(18,30)">
        <ProcessLane name="Process A" lanes={lanes} pos={a} x={0} />
        <ProcessLane name="Process B" lanes={lanes} pos={b} x={124} />
      </g>
      <g transform={`translate(${w / 2},${h - 10})`}>
        <text textAnchor="middle" fontSize={10} fontFamily="var(--mono)"
          fill={lock || peterson ? ACCENT : 'var(--t2)'}>{vars}</text>
      </g>
    </svg>
  );
};

/* ---------------- River crossing: two banks ---------------- */

const ICON: Record<string, string> = { F: '🧑', W: '🐺', G: '🐐', C: '🥬', M: '🐍' };

export const RiverSchematic: React.FC<{ items: string[]; far: Record<string, 1 | 0>; farmerFar: boolean }> = ({ items, far, farmerFar }) => {
  const w = 300, h = 120;
  const bankW = 118, gap = w - 2 * bankW;
  const draw = (onFar: boolean, x: number, title: string) => {
    const here = items.filter((it) => (it === 'F' ? farmerFar === onFar : far[it] === (onFar ? 1 : 0)));
    return (
      <g transform={`translate(${x},0)`}>
        <rect x={0} y={20} width={bankW} height={h - 28} rx={10} fill="rgba(52,211,153,.07)" stroke="var(--border)" />
        <text x={bankW / 2} y={14} textAnchor="middle" fontSize={9.5} fontFamily="var(--mono)" fill="var(--t2)">{title}</text>
        {here.map((it, i) => (
          <text key={it} x={18 + (i % 3) * 34} y={48 + Math.floor(i / 3) * 34} fontSize={22}>{ICON[it] ?? it}</text>
        ))}
      </g>
    );
  };
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      {draw(false, 0, 'NEAR BANK')}
      {/* river */}
      <g transform={`translate(${bankW},0)`}>
        <rect x={0} y={20} width={gap} height={h - 28} fill="rgba(56,189,248,.10)" />
        <text x={gap / 2} y={h / 2 + 4} textAnchor="middle" fontSize={16}>{farmerFar ? '➡' : '⬅'}</text>
      </g>
      {draw(true, bankW + gap, 'FAR BANK')}
    </svg>
  );
};

/* ---------------- shared: state-space graph with a floating schematic ---------------- */

export const StateSpace: React.FC<{
  nodes: GNode[]; edges: GEdge[]; width: number; height: number; radius: number;
  schematic?: React.ReactNode;
}> = ({ nodes, edges, width, height, radius, schematic }) => (
  <div style={{ position: 'relative', display: 'inline-block' }}>
    <GraphCanvas width={width} height={height} radius={radius} nodes={nodes} edges={edges} />
    {schematic && (
      <div style={{ position: 'absolute', top: 10, right: 10, pointerEvents: 'none' }}>{schematic}</div>
    )}
  </div>
);
