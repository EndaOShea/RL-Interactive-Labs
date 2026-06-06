import React from 'react';
import { ACC } from '../../stage/primitives';

// Dendrogram (merge tree) for hierarchical clustering. Leaves sit at the bottom;
// each internal node's height is the distance at which its two children merged.
export type DLeaf = { id: number; color?: string };
export type DInternal = { height: number; left: DendroNode; right: DendroNode };
export type DendroNode = DLeaf | DInternal;
const isLeaf = (n: DendroNode): n is DLeaf => !('height' in n);

export interface DendrogramProps {
  root: DendroNode | null;
  maxHeight: number;
  leafCount: number;
  cut?: number;
  leafColor?: (id: number) => string;
  width?: number;
  height?: number;
}

const Dendrogram: React.FC<DendrogramProps> = ({ root, maxHeight, leafCount, cut, leafColor, width = 540, height = 440 }) => {
  const padL = 14, padR = 14, padT = 18, padB = 22;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const xOf = (i: number) => (leafCount <= 1 ? padL + plotW / 2 : padL + (i / (leafCount - 1)) * plotW);
  const yOf = (h: number) => padT + (1 - (maxHeight ? h / maxHeight : 0)) * plotH;

  let leafIdx = 0;
  const segs: { x1: number; y1: number; x2: number; y2: number }[] = [];
  const leaves: { x: number; id: number; color?: string }[] = [];

  const layout = (n: DendroNode): { x: number; y: number } => {
    if (isLeaf(n)) { const x = xOf(leafIdx++); leaves.push({ x, id: n.id, color: n.color }); return { x, y: yOf(0) }; }
    const L = layout(n.left), R = layout(n.right);
    const y = yOf(n.height);
    segs.push({ x1: L.x, y1: L.y, x2: L.x, y2: y });
    segs.push({ x1: R.x, y1: R.y, x2: R.x, y2: y });
    segs.push({ x1: L.x, y1: y, x2: R.x, y2: y });
    return { x: (L.x + R.x) / 2, y };
  };
  if (root) layout(root);
  const cutY = cut != null ? yOf(cut) : null;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', borderRadius: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      {segs.map((s, i) => (
        <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} stroke="var(--t1)" strokeWidth={1.6} strokeLinecap="round" />
      ))}
      {cutY != null && (
        <g>
          <line x1={padL} y1={cutY} x2={width - padR} y2={cutY} stroke={ACC} strokeWidth={1.6} strokeDasharray="6 5" />
          <text x={width - padR} y={cutY - 5} textAnchor="end" fontSize="9.5" fontFamily="var(--mono)" fill={ACC}>cut</text>
        </g>
      )}
      {leaves.map((l, i) => (
        <circle key={i} cx={l.x} cy={yOf(0)} r={4.5} fill={leafColor ? leafColor(l.id) : (l.color || 'var(--t2)')} stroke="rgba(8,11,20,.6)" strokeWidth={0.8} />
      ))}
      <text x={padL} y={12} fontSize="9.5" fontFamily="var(--mono)" fill="var(--t2)">merge distance ↑</text>
    </svg>
  );
};

export default Dendrogram;
