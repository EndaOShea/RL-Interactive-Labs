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

const LANE_NAMES = ['Idle', 'Wait', 'Critical'];
const LANE_COLORS = ['#64748b', '#38bdf8', '#fb7185'];

const ProcessLane: React.FC<{ name: string; pos: number; x: number }> = ({ name, pos, x }) => {
  const isLight = useTheme() === 'light';
  const cellW = 78, cellH = 26, gap = 6, top = 0;
  return (
    <g transform={`translate(${x},0)`}>
      <text x={0} y={top - 8} fontSize={11} fontFamily="var(--mono)" fill="var(--t1)" fontWeight={600}>{name}</text>
      {LANE_NAMES.map((ln, i) => {
        const y = top + i * (cellH + gap);
        const here = pos === i;
        return (
          <g key={i}>
            <rect x={0} y={y} width={cellW} height={cellH} rx={6}
              fill={here ? LANE_COLORS[i] : (isLight ? 'rgba(50,60,90,.10)' : 'rgba(120,130,170,.10)')}
              stroke={here ? LANE_COLORS[i] : 'var(--border)'} strokeWidth={here ? 1.6 : 1}
              style={here ? { filter: `drop-shadow(0 0 6px ${LANE_COLORS[i]})` } : undefined} />
            <text x={cellW / 2} y={y + cellH / 2 + 4} textAnchor="middle" fontSize={11}
              fontFamily="var(--disp)" fontWeight={600}
              fill={here ? 'rgba(8,11,20,.9)' : 'var(--t2)'}>{ln}</text>
          </g>
        );
      })}
    </g>
  );
};

export const MutexSchematic: React.FC<{ a: number; b: number; lock: boolean; unsafe: boolean }> = ({ a, b, lock, unsafe }) => {
  const isLight = useTheme() === 'light';
  const w = 240, h = 132;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <rect x={0} y={0} width={w} height={h} rx={12} fill={isLight ? 'var(--bg2)' : PANEL}
        stroke={unsafe ? (isLight ? 'var(--bad)' : '#f87171') : 'var(--border)'} strokeWidth={unsafe ? 1.8 : 1} />
      <g transform="translate(18,30)">
        <ProcessLane name="Process A" pos={a} x={0} />
        <ProcessLane name="Process B" pos={b} x={124} />
      </g>
      <g transform={`translate(${w / 2},${h - 12})`}>
        <text textAnchor="middle" fontSize={10} fontFamily="var(--mono)"
          fill={lock ? ACCENT : 'var(--t2)'}>{lock ? '🔒 lock held' : 'lock free'}</text>
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
