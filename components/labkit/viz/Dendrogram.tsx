import React from 'react';
import { ACC } from '../../stage/primitives';
import { useTheme } from '../../../utils/theme';

// Dendrogram (merge tree) for hierarchical clustering. Leaves sit at the bottom;
// each internal node's height is the distance at which its two children merged.
// Leaf order comes from the full tree, so the leaves never move while the tree is
// built: with `visibleSteps` only merges whose `step` is below it are drawn (a
// forest of the clusters formed so far). A merge lower than one of its children
// (an inversion, possible with centroid linkage) is drawn as it is — its bar sits
// below the child's top.
export type DLeaf = { id: number; color?: string };
export type DInternal = { height: number; left: DendroNode; right: DendroNode; step?: number };
export type DendroNode = DLeaf | DInternal;
const isLeaf = (n: DendroNode): n is DLeaf => !('height' in n);

export interface DendrogramProps {
  root: DendroNode | null;
  maxHeight: number;
  leafCount: number;
  cut?: number;
  cutLabel?: string;
  /** Draw only internal nodes with step < visibleSteps (default: all). */
  visibleSteps?: number;
  leafColor?: (id: number) => string;
  /** Colour for a merge's links (e.g. its flat cluster when it lies below the cut). */
  linkColor?: (node: DInternal) => string | undefined;
  width?: number;
  height?: number;
}

const Dendrogram: React.FC<DendrogramProps> = ({ root, maxHeight, leafCount, cut, cutLabel, visibleSteps, leafColor, linkColor, width = 540, height = 440 }) => {
  const isLight = useTheme() === 'light';
  const padL = 40, padR = 14, padT = 22, padB = 22;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const top = maxHeight > 0 ? maxHeight * 1.06 : 1;
  const xOf = (i: number) => (leafCount <= 1 ? padL + plotW / 2 : padL + (i / (leafCount - 1)) * plotW);
  const yOf = (h: number) => padT + (1 - Math.min(h, top) / top) * plotH;

  let leafIdx = 0;
  const segs: { x1: number; y1: number; x2: number; y2: number; color?: string }[] = [];
  const leaves: { x: number; id: number; color?: string }[] = [];

  const layout = (n: DendroNode): { x: number; y: number } => {
    if (isLeaf(n)) { const x = xOf(leafIdx++); leaves.push({ x, id: n.id, color: n.color }); return { x, y: yOf(0) }; }
    const L = layout(n.left), R = layout(n.right);
    const y = yOf(n.height);
    const visible = visibleSteps == null || n.step == null || n.step < visibleSteps;
    if (visible) {
      const color = linkColor?.(n);
      segs.push({ x1: L.x, y1: L.y, x2: L.x, y2: y, color });
      segs.push({ x1: R.x, y1: R.y, x2: R.x, y2: y, color });
      segs.push({ x1: L.x, y1: y, x2: R.x, y2: y, color });
    }
    return { x: (L.x + R.x) / 2, y };
  };
  if (root) layout(root);
  const cutY = cut != null ? yOf(cut) : null;
  const ticks = maxHeight > 0 ? [0, maxHeight / 2, maxHeight] : [0];

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={padL - 4} y1={yOf(t)} x2={padL + plotW} y2={yOf(t)} stroke={isLight ? 'rgba(50,60,90,.10)' : 'rgba(120,130,170,.08)'} />
          <text x={padL - 6} y={yOf(t) + 3} textAnchor="end" fontSize="8.5" fontFamily="var(--mono)" fill="var(--t2)">{t.toFixed(t >= 1 ? 2 : 3)}</text>
        </g>
      ))}
      {segs.map((s, i) => (
        <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} stroke={s.color || 'var(--t1)'} strokeWidth={1.6} strokeLinecap="round" />
      ))}
      {cutY != null && (
        <g>
          <line x1={padL} y1={cutY} x2={width - padR} y2={cutY} stroke={ACC} strokeWidth={1.6} strokeDasharray="6 5" />
          <text x={width - padR} y={cutY - 5} textAnchor="end" fontSize="9.5" fontFamily="var(--mono)" fill={ACC}>{cutLabel ?? 'cut'}</text>
        </g>
      )}
      {leaves.map((l, i) => (
        <circle key={i} cx={l.x} cy={yOf(0)} r={4.5} fill={leafColor ? leafColor(l.id) : (l.color || 'var(--t2)')} stroke={isLight ? 'rgba(255,255,255,.85)' : 'rgba(8,11,20,.6)'} strokeWidth={0.8} />
      ))}
      <text x={padL} y={13} fontSize="9.5" fontFamily="var(--mono)" fill="var(--t2)">merge distance ↑</text>
    </svg>
  );
};

export default Dendrogram;
