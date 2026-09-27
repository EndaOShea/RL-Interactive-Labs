// CART decision tree for the Decision Tree lab (pure maths, no React). Binary
// classes, axis-aligned splits at midpoints between consecutive distinct values,
// routing x ≤ thr → left. A node becomes a leaf when it is pure, at max depth,
// holds fewer than MIN_SPLIT (or 2·minLeaf) points, or no split lowers impurity.
// Ties: the first best split wins (feature x₁ before x₂, ascending thresholds);
// a leaf predicts its majority class (ties → class 0).

export type Crit = 'gini' | 'entropy';
export interface TPt { x: number; y: number; cls: number; }

/** Nodes with fewer points than this are never split (sklearn min_samples_split). */
export const MIN_SPLIT = 4;

export type TNode =
  | { leaf: true; cls: number; n: number; imp: number; counts: [number, number] }
  | { leaf: false; feat: 0 | 1; thr: number; imp: number; n: number; gain: number; counts: [number, number]; left: TNode; right: TNode };

const fv = (p: TPt, f: 0 | 1) => (f === 0 ? p.x : p.y);

export function countsOf(pts: readonly TPt[]): [number, number] {
  let c0 = 0, c1 = 0;
  for (const p of pts) { if (p.cls === 0) c0++; else c1++; }
  return [c0, c1];
}

/** Gini 1 − Σp² or entropy −Σ p log₂ p of a class-count vector (classes in order 0, 1). */
export function impurityOf(c: readonly number[], n: number, crit: Crit): number {
  if (n === 0) return 0;
  let v = crit === 'gini' ? 1 : 0;
  for (const k of c) {
    if (k === 0) continue;
    const p = k / n;
    if (crit === 'gini') v -= p * p; else v -= p * Math.log2(p);
  }
  return v;
}

interface Best { feat: 0 | 1; thr: number; score: number; }

/** All candidate splits of a node with their weighted child impurity (for plots). */
export function candidateSplits(pts: readonly TPt[], crit: Crit, minLeaf: number): { feat: 0 | 1; thr: number; score: number; gain: number }[] {
  const n = pts.length;
  const imp = impurityOf(countsOf(pts), n, crit);
  const out: { feat: 0 | 1; thr: number; score: number; gain: number }[] = [];
  for (const feat of [0, 1] as const) {
    const vals = [...new Set(pts.map((p) => fv(p, feat)))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i]! + vals[i + 1]!) / 2;
      let l0 = 0, l1 = 0, r0 = 0, r1 = 0;
      for (const p of pts) {
        if (fv(p, feat) <= thr) { if (p.cls === 0) l0++; else l1++; } else if (p.cls === 0) r0++; else r1++;
      }
      const nl = l0 + l1, nr = r0 + r1;
      if (nl < minLeaf || nr < minLeaf) continue;
      const score = (nl * impurityOf([l0, l1], nl, crit) + nr * impurityOf([r0, r1], nr, crit)) / n;
      out.push({ feat, thr, score, gain: imp - score });
    }
  }
  return out;
}

export function buildTree(pts: readonly TPt[], depth: number, maxDepth: number, crit: Crit, minLeaf: number): TNode {
  const n = pts.length;
  const counts = countsOf(pts);
  const imp = impurityOf(counts, n, crit);
  const maj = counts[1] > counts[0] ? 1 : 0;
  if (depth >= maxDepth || imp < 1e-9 || n < Math.max(MIN_SPLIT, 2 * minLeaf)) return { leaf: true, cls: maj, n, imp, counts };
  let best: Best | null = null;
  for (const c of candidateSplits(pts, crit, minLeaf)) {
    if (!best || c.score < best.score) best = { feat: c.feat, thr: c.thr, score: c.score };
  }
  if (!best || best.score >= imp - 1e-9) return { leaf: true, cls: maj, n, imp, counts };
  const f = best.feat, t = best.thr;
  const L = pts.filter((p) => fv(p, f) <= t), R = pts.filter((p) => fv(p, f) > t);
  return {
    leaf: false, feat: f, thr: t, imp, n, gain: imp - best.score, counts,
    left: buildTree(L, depth + 1, maxDepth, crit, minLeaf), right: buildTree(R, depth + 1, maxDepth, crit, minLeaf),
  };
}

export const classifyTree = (n: TNode, x: number, y: number): number =>
  n.leaf ? n.cls : classifyTree((n.feat === 0 ? x : y) <= n.thr ? n.left : n.right, x, y);

export function countNodes(n: TNode): [number, number] {
  if (n.leaf) return [1, 1];
  const [ln, ll] = countNodes(n.left), [rn, rl] = countNodes(n.right);
  return [1 + ln + rn, ll + rl];
}

export function treeDepth(n: TNode): number {
  return n.leaf ? 0 : 1 + Math.max(treeDepth(n.left), treeDepth(n.right));
}

export function accuracy(t: TNode, pts: readonly TPt[]): number {
  if (!pts.length) return 0;
  let ok = 0;
  for (const p of pts) if (classifyTree(t, p.x, p.y) === p.cls) ok++;
  return ok / pts.length;
}

/** The splits whose node sits at depth `d` (0 = root), with their impurity decrease. */
export function splitsAtDepth(n: TNode, d: number, depth = 0, acc: { feat: 0 | 1; thr: number; gain: number }[] = []) {
  if (n.leaf) return acc;
  if (depth === d) acc.push({ feat: n.feat, thr: n.thr, gain: n.gain });
  else { splitsAtDepth(n.left, d, depth + 1, acc); splitsAtDepth(n.right, d, depth + 1, acc); }
  return acc;
}

/** Exact split segments inside [0,1]² (each internal node's cut, clipped to its cell). */
export function splitSegments(n: TNode, box = { x0: 0, x1: 1, y0: 0, y1: 1 }, depth = 0, out: { x1: number; y1: number; x2: number; y2: number; depth: number }[] = []) {
  if (n.leaf) return out;
  if (n.feat === 0) {
    out.push({ x1: n.thr, y1: box.y0, x2: n.thr, y2: box.y1, depth });
    splitSegments(n.left, { ...box, x1: n.thr }, depth + 1, out);
    splitSegments(n.right, { ...box, x0: n.thr }, depth + 1, out);
  } else {
    out.push({ x1: box.x0, y1: n.thr, x2: box.x1, y2: n.thr, depth });
    splitSegments(n.left, { ...box, y1: n.thr }, depth + 1, out);
    splitSegments(n.right, { ...box, y0: n.thr }, depth + 1, out);
  }
  return out;
}
