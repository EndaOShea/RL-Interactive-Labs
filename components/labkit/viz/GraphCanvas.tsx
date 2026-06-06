import React from 'react';

// Node/edge graph (SVG). Powers graph search (BFS/Dijkstra/A*) and is reusable
// for future trees/automata/Bayes-net labs. Coordinates are in [0,1].
export type NodeState = 'idle' | 'frontier' | 'visited' | 'path' | 'current' | 'start' | 'goal';

const NODE_COLORS: Record<NodeState, string> = {
  idle: '#2a3350', start: '#34d399', goal: '#f87171', frontier: '#38bdf8',
  visited: '#1e3a52', path: '#fbbf24', current: '#ffffff',
};

export interface GNode { id: string; x: number; y: number; label?: string; sub?: string; state?: NodeState; color?: string; }
export interface GEdge { from: string; to: string; weight?: number; state?: 'idle' | 'active' | 'path'; }

export interface GraphCanvasProps {
  nodes: GNode[];
  edges: GEdge[];
  width?: number;
  height?: number;
  radius?: number;
}

const GraphCanvas: React.FC<GraphCanvasProps> = ({ nodes, edges, width = 560, height = 460, radius = 17 }) => {
  const pad = radius + 14;
  const sx = (x: number) => pad + x * (width - 2 * pad);
  const sy = (y: number) => pad + y * (height - 2 * pad);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edgeColor = (s?: string) => (s === 'path' ? '#fbbf24' : s === 'active' ? '#38bdf8' : 'rgba(120,130,170,.28)');

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', borderRadius: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      {/* edges */}
      {edges.map((e, i) => {
        const a = byId.get(e.from), b = byId.get(e.to);
        if (!a || !b) return null;
        const x1 = sx(a.x), y1 = sy(a.y), x2 = sx(b.x), y2 = sy(b.y);
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={edgeColor(e.state)} strokeWidth={e.state === 'path' ? 3.4 : e.state === 'active' ? 2.4 : 1.6} strokeLinecap="round" />
            {e.weight != null && (
              <g>
                <rect x={mx - 10} y={my - 9} width={20} height={16} rx={4} fill="var(--bg0)" stroke="var(--border)" />
                <text x={mx} y={my + 3} textAnchor="middle" fontSize="10" fontFamily="var(--mono)" fill="var(--t1)">{e.weight}</text>
              </g>
            )}
          </g>
        );
      })}
      {/* nodes */}
      {nodes.map((n) => {
        const st = n.state || 'idle';
        const color = n.color || NODE_COLORS[st];
        const cx = sx(n.x), cy = sy(n.y);
        const glow = st === 'start' || st === 'goal' || st === 'current' || st === 'path';
        const darkText = st === 'current' || st === 'path' || st === 'start' || st === 'goal' || st === 'frontier';
        return (
          <g key={n.id}>
            <circle cx={cx} cy={cy} r={radius} fill={color} stroke="rgba(8,11,20,.55)" strokeWidth={1.5} style={glow ? { filter: `drop-shadow(0 0 7px ${color})` } : undefined} />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="12" fontWeight={600} fontFamily="var(--disp)" fill={darkText ? 'rgba(8,11,20,.85)' : 'var(--t0)'}>{n.label ?? n.id}</text>
            {n.sub && <text x={cx} y={cy + radius + 13} textAnchor="middle" fontSize="9.5" fontFamily="var(--mono)" fill="var(--t2)">{n.sub}</text>}
          </g>
        );
      })}
    </svg>
  );
};

export default GraphCanvas;
