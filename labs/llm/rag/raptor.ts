// labs/llm/rag/raptor.ts — RAPTOR's recursive summary tree, COMPUTED from the
// chunks (no hand-assigned clusters, no hand-written summaries):
//   • cluster the current layer's vectors with k-means (deterministic maximin
//     initialisation), choosing k ∈ [2, min(6, n−1)] by the best mean silhouette;
//   • each cluster becomes a summary node whose text is EXTRACTIVE — the two
//     self-contained sentences of its children nearest the cluster centroid —
//     and whose vector is the embedding of that summary text; its label names
//     the centroid's strongest topic axes;
//   • recurse while a layer has more than RAPTOR_MAX_K nodes, then put one
//     root over the last layer.
// Retrieval is RAPTOR's "collapsed tree": every node, leaf or summary, is scored
// against the query by cosine and the top-k are kept.
import { AXES, embedText, cosine, sentences, l2norm, selfContained } from './corpus';
import type { Chunk } from './retrieval';

export interface TreeNode { id: string; level: number; label: string; text: string; childIds: string[]; vec: number[]; titles: string[]; }

export const RAPTOR_MAX_K = 6, SUMMARY_SENTENCES = 2;

function dist2(a: number[], b: number[]): number {
  let s = 0; for (let i = 0; i < a.length; i++) { const d = (a[i] ?? 0) - (b[i] ?? 0); s += d * d; } return s;
}
function meanOf(vs: number[][], dim: number): number[] {
  const m = new Array<number>(dim).fill(0);
  vs.forEach((v) => v.forEach((x, i) => { m[i] = (m[i] ?? 0) + x; }));
  return m.map((x) => x / Math.max(1, vs.length));
}

// k-means with maximin init: the first centre is the point farthest from the
// mean, each next centre the point farthest from its nearest centre (ties →
// lowest index); Lloyd iterations until the assignment stops changing.
export function kmeans(X: number[][], k: number): number[] {
  const n = X.length, dim = X[0]?.length ?? 0;
  const mean = meanOf(X, dim);
  const centres: number[][] = [];
  let first = 0, far = -1;
  X.forEach((x, i) => { const d = dist2(x, mean); if (d > far) { far = d; first = i; } });
  centres.push([...(X[first] ?? [])]);
  while (centres.length < Math.min(k, n)) {
    let pick = 0, best = -1;
    X.forEach((x, i) => { const d = Math.min(...centres.map((c) => dist2(x, c))); if (d > best) { best = d; pick = i; } });
    centres.push([...(X[pick] ?? [])]);
  }
  let assign = new Array<number>(n).fill(-1);
  for (let it = 0; it < 100; it++) {
    const next = X.map((x) => { let bc = 0, bd = Infinity; centres.forEach((c, ci) => { const d = dist2(x, c); if (d < bd) { bd = d; bc = ci; } }); return bc; });
    const changed = next.some((a, i) => a !== assign[i]);
    assign = next;
    centres.forEach((_, ci) => {
      const mem = X.filter((_, i) => assign[i] === ci);
      if (mem.length) centres[ci] = meanOf(mem, dim);
    });
    if (!changed) break;
  }
  return assign;
}

// mean silhouette (Euclidean): s(i) = (b − a) / max(a, b); a singleton scores 0.
export function silhouette(X: number[][], assign: number[]): number {
  const n = X.length; if (!n) return 0;
  const ids = [...new Set(assign)];
  let total = 0;
  X.forEach((x, i) => {
    const own = assign[i];
    const same = X.filter((_, j) => j !== i && assign[j] === own);
    if (!same.length) return;
    const a = same.reduce((s, y) => s + Math.sqrt(dist2(x, y)), 0) / same.length;
    let b = Infinity;
    ids.forEach((c) => {
      if (c === own) return;
      const other = X.filter((_, j) => assign[j] === c);
      if (other.length) b = Math.min(b, other.reduce((s, y) => s + Math.sqrt(dist2(x, y)), 0) / other.length);
    });
    const den = Math.max(a, b);
    if (isFinite(b) && den > 0) total += (b - a) / den;
  });
  return total / n;
}

export function chooseK(X: number[][]): { k: number; assign: number[]; scores: { k: number; s: number }[] } {
  const scores: { k: number; s: number }[] = [];
  let bestK = 1, bestS = -Infinity, bestA = X.map(() => 0);
  for (let k = 2; k <= Math.min(RAPTOR_MAX_K, X.length - 1); k++) {
    const a = kmeans(X, k); const s = silhouette(X, a);
    scores.push({ k, s });
    if (s > bestS) { bestS = s; bestK = k; bestA = a; }
  }
  return { k: bestK, assign: bestA, scores };
}

// the centroid's strongest topic axes (those ≥ half the largest), at most two
export function axisLabel(v: number[]): string {
  const ranked = AXES.map((a, i) => ({ a, w: v[i] ?? 0 })).sort((x, y) => y.w - x.w);
  const top = ranked[0]?.w ?? 0;
  if (top <= 0) return 'no topic';
  return ranked.filter((r) => r.w >= 0.5 * top).slice(0, 2).map((r) => r.a).join(' · ');
}

function summarize(children: TreeNode[], id: string, level: number): TreeNode {
  const dim = children[0]?.vec.length ?? 0;
  const centre = l2norm(meanOf(children.map((c) => c.vec), dim));
  const seen = new Set<string>();
  const all: { text: string; score: number; k: number; ok: boolean }[] = [];
  children.forEach((ch) => sentences(ch.text).forEach((s) => {
    if (seen.has(s)) return; seen.add(s);
    all.push({ text: s, score: cosine(embedText(s), centre), k: all.length, ok: selfContained(s) });
  }));
  const cands = all.some((c) => c.ok) ? all.filter((c) => c.ok) : all;
  const top = [...cands].sort((a, b) => b.score - a.score).slice(0, SUMMARY_SENTENCES).sort((a, b) => a.k - b.k);
  const text = top.map((c) => c.text).join(' ');
  const titles: string[] = [];
  children.forEach((ch) => ch.titles.forEach((t) => { if (!titles.includes(t)) titles.push(t); }));
  return { id, level, label: axisLabel(centre), text, childIds: children.map((c) => c.id), vec: embedText(text), titles };
}

export function buildTree(chunks: Chunk[]): TreeNode[] {
  const nodes: TreeNode[] = chunks.map((c) => ({ id: c.id, level: 0, label: c.title, text: c.text, childIds: [], vec: c.vec, titles: [c.title] }));
  let layer = nodes.slice(); let level = 0;
  while (layer.length > RAPTOR_MAX_K) {
    const { assign } = chooseK(layer.map((n) => n.vec));
    const groups: TreeNode[][] = [];
    const slot = new Map<number, number>();
    layer.forEach((node, i) => {
      const c = assign[i] ?? 0;
      if (!slot.has(c)) { slot.set(c, groups.length); groups.push([]); }
      groups[slot.get(c) ?? 0]?.push(node);
    });
    if (groups.length <= 1 || groups.length >= layer.length) break; // no reduction → stop and root it
    level++;
    const next = groups.map((g, gi) => summarize(g, `L${level}.${gi}`, level));
    nodes.push(...next);
    layer = next;
  }
  if (layer.length) nodes.push(summarize(layer, 'root', level + 1)); // childIds only ever name existing nodes
  return nodes;
}

// Collapsed-tree retrieval: flat cosine over EVERY node, no traversal.
export function retrieveTree(query: string, tree: TreeNode[], k: number): { id: string; score: number }[] {
  const q = embedText(query);
  return tree.map((n) => ({ id: n.id, score: cosine(q, n.vec) })).sort((a, b) => b.score - a.score).slice(0, k);
}
