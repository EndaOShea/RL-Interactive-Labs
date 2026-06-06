import React, { useEffect, useRef } from 'react';

// Grid board for search/pathfinding. Each cell has a discrete state with its own
// colour; supports drag-to-paint walls. Pure/presentational — the lab owns the
// grid + search state. Reusable for mazes and model-checking grids.
export type CellState = 'empty' | 'wall' | 'start' | 'goal' | 'frontier' | 'visited' | 'path' | 'current';

const COLORS: Record<CellState, string> = {
  empty: 'rgba(20,26,44,.55)',
  wall: '#070b14',
  start: '#34d399',
  goal: '#f87171',
  frontier: '#38bdf8',
  visited: 'rgba(56,189,248,.22)',
  path: '#fbbf24',
  current: '#ffffff',
};

export interface GridBoardProps {
  cols: number;
  rows: number;
  state: (idx: number) => CellState;
  label?: (idx: number) => string | undefined;
  cell?: number;
  gap?: number;
  onPaint?: (idx: number, mode: 'add' | 'remove') => void;
}

const GridBoard: React.FC<GridBoardProps> = ({ cols, rows, state, label, cell = 28, gap = 3, onPaint }) => {
  const painting = useRef(false);
  const mode = useRef<'add' | 'remove'>('add');
  const W = cols * (cell + gap) + gap;
  const H = rows * (cell + gap) + gap;

  useEffect(() => {
    const up = () => { painting.current = false; };
    window.addEventListener('pointerup', up);
    return () => window.removeEventListener('pointerup', up);
  }, []);

  const down = (idx: number) => {
    if (!onPaint) return;
    mode.current = state(idx) === 'wall' ? 'remove' : 'add';
    painting.current = true;
    onPaint(idx, mode.current);
  };
  const enter = (idx: number) => {
    if (!onPaint || !painting.current) return;
    const s = state(idx);
    if (s === 'start' || s === 'goal') return;
    onPaint(idx, mode.current);
  };

  return (
    <svg
      width={W} height={H} viewBox={`0 0 ${W} ${H}`}
      style={{ display: 'block', borderRadius: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%', maxHeight: '70vh', touchAction: 'none' }}
    >
      {Array.from({ length: cols * rows }, (_, idx) => {
        const r = Math.floor(idx / cols), c = idx % cols;
        const s = state(idx);
        const x = gap + c * (cell + gap), y = gap + r * (cell + gap);
        const lab = label?.(idx);
        const glow = s === 'start' || s === 'goal' || s === 'current' || s === 'path';
        return (
          <g key={idx}>
            <rect
              x={x} y={y} width={cell} height={cell} rx={4}
              fill={COLORS[s]}
              stroke={s === 'empty' ? 'rgba(120,130,170,.10)' : s === 'wall' ? 'rgba(120,130,170,.06)' : 'rgba(8,11,20,.4)'}
              strokeWidth={1}
              style={glow ? { filter: `drop-shadow(0 0 6px ${COLORS[s]})` } : undefined}
              onPointerDown={() => down(idx)}
              onPointerEnter={() => enter(idx)}
              cursor={onPaint ? 'pointer' : 'default'}
            />
            {lab && cell >= 22 && (
              <text x={x + cell / 2} y={y + cell / 2 + 3} textAnchor="middle" fontSize={Math.min(10, cell / 2.6)} fontFamily="var(--mono)" fill={s === 'visited' || s === 'empty' ? 'var(--t2)' : 'rgba(8,11,20,.8)'} pointerEvents="none">{lab}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

export default GridBoard;
