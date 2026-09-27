// Agglomerative (bottom-up) hierarchical clustering for the Hierarchical lab.
// Pure maths, no React.
//
// The merge loop is the textbook O(n³) greedy one: repeatedly join the two
// clusters with the smallest linkage distance. Ward's height is √ΔSSE (the
// increase in within-cluster sum of squares); SciPy reports √(2·ΔSSE), which
// the Python export converts. Centroid linkage can produce "inversions" (a merge
// lower than one of its children), so every "cut at a height" here follows
// SciPy's fcluster(criterion='distance'): a subtree stays one flat cluster iff
// the HIGHEST merge inside it is ≤ the cut.
import type { UPt } from './shared';

export type Linkage = 'single' | 'complete' | 'average' | 'ward' | 'centroid';

export interface ALeaf { id: number }
export interface AInternal {
  height: number;
  left: ANode;
  right: ANode;
  /** Merge index (0 = first merge). */
  step: number;
  /** Highest merge height anywhere in this subtree (≥ height; > height only after an inversion). */
  maxH: number;
  /** Any leaf in the subtree (for colouring). */
  leaf: number;
}
export type ANode = ALeaf | AInternal;

export interface Merge {
  /** SciPy-style ids of the two merged clusters (leaves 0..n−1, merge m creates id n+m). */
  a: number;
  b: number;
  height: number;
  maxH: number;
  size: number;
  members: number[];
}

const centroidOf = (pts: UPt[], m: number[]) => {
  let x = 0, y = 0; for (const i of m) { x += pts[i]!.x; y += pts[i]!.y; } return { x: x / m.length, y: y / m.length };
};

export function agglomerate(pts: UPt[], linkage: Linkage) {
  const n = pts.length;
  const D = pts.map((p) => pts.map((q) => Math.hypot(p.x - q.x, p.y - q.y)));
  let clusters: { id: number; members: number[]; node: ANode }[] = pts.map((_, i) => ({ id: i, members: [i], node: { id: i } }));
  const merges: Merge[] = [];
  const cd = (A: number[], B: number[]) => {
    if (linkage === 'ward') {
      // Ward: √ΔSSE, where ΔSSE = |A||B|/(|A|+|B|)·‖c_A − c_B‖² is the within-cluster SS increase.
      const ca = centroidOf(pts, A), cb = centroidOf(pts, B);
      const d2 = (ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2;
      return Math.sqrt((A.length * B.length / (A.length + B.length)) * d2);
    }
    if (linkage === 'centroid') {
      const ca = centroidOf(pts, A), cb = centroidOf(pts, B);
      return Math.hypot(ca.x - cb.x, ca.y - cb.y);
    }
    let agg = linkage === 'single' ? Infinity : linkage === 'complete' ? -Infinity : 0; let cnt = 0;
    for (const a of A) for (const b of B) {
      const d = D[a]![b]!;
      if (linkage === 'single') agg = Math.min(agg, d); else if (linkage === 'complete') agg = Math.max(agg, d); else { agg += d; cnt++; }
    }
    return linkage === 'average' ? agg / cnt : agg;
  };
  const maxOf = (nd: ANode) => ('height' in nd ? nd.maxH : 0);
  while (clusters.length > 1) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const d = cd(clusters[i]!.members, clusters[j]!.members);
        if (d < bd) { bd = d; bi = i; bj = j; }
      }
    }
    const A = clusters[bi]!, B = clusters[bj]!;
    const step = merges.length;
    const maxH = Math.max(bd, maxOf(A.node), maxOf(B.node));
    const node: AInternal = { height: bd, left: A.node, right: B.node, step, maxH, leaf: A.members[0]! };
    const members = [...A.members, ...B.members];
    merges.push({ a: A.id, b: B.id, height: bd, maxH, size: members.length, members });
    clusters = clusters.filter((_, k) => k !== bi && k !== bj);
    clusters.push({ id: n + step, members, node });
  }
  const maxHeight = merges.reduce((m, g) => Math.max(m, g.height), 0);
  return { root: clusters[0]?.node ?? null, merges, maxHeight };
}

/**
 * Flat clusters after the first `done` merges, cut at height `h` (SciPy
 * fcluster 'distance' semantics). Returns a label per point, numbered in order
 * of each cluster's lowest point index, and the cluster count.
 */
export function flatClusters(n: number, merges: Merge[], done: number, h: number) {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => { while (parent[i] !== i) { parent[i] = parent[parent[i]!]!; i = parent[i]!; } return i; };
  const m = Math.min(done, merges.length);
  for (let s = 0; s < m; s++) {
    const g = merges[s]!;
    if (g.maxH > h) continue; // this merge (and every merge above it) is cut
    const r0 = find(g.members[0]!);
    for (const i of g.members) { const r = find(i); if (r !== r0) parent[r] = r0; }
  }
  const ids = new Map<number, number>();
  const labels = Array.from({ length: n }, (_, i) => {
    const r = find(i);
    if (!ids.has(r)) ids.set(r, ids.size);
    return ids.get(r)!;
  });
  return { labels, k: ids.size };
}

/**
 * The height that reproduces the partition after `done` merges: below the first
 * merge at the start, above the root at the end, otherwise between the highest
 * merge so far and the next merge (on the highest merge so far when the next one
 * is an inversion).
 */
export function autoCut(merges: Merge[], done: number, maxHeight: number) {
  if (!merges.length) return 0;
  if (done <= 0) return merges[0]!.maxH / 2;
  if (done >= merges.length) return maxHeight * 1.04;
  let M = 0;
  for (let s = 0; s < done; s++) M = Math.max(M, merges[s]!.maxH);
  const next = merges[done]!.maxH;
  return next > M ? (M + next) / 2 : M;
}

/** Midpoint of the largest gap between consecutive merge levels (≥ 2 clusters) — the classic "natural" cut. */
export function largestGapCut(merges: Merge[]) {
  const H = merges.map((g) => g.maxH).sort((a, b) => a - b);
  let best = 1, gap = -1;
  for (let i = 1; i < H.length; i++) { const g = H[i]! - H[i - 1]!; if (g > gap) { gap = g; best = i; } }
  return H.length > 1 ? (H[best - 1]! + H[best]!) / 2 : (H[0] ?? 0) / 2;
}

/** Merges lower than one of their children (only centroid linkage can do this). */
export const inversionCount = (merges: Merge[]) => merges.filter((g) => g.height < g.maxH).length;
