// Second-order (Newton) gradient boosting for the Gradient Boosting lab — pure
// maths, no React. Logistic loss: g = p − y, h = p(1 − p) (floored at 1e-6).
// Leaf weight w* = −ΣG/(ΣH + λ) (0 for an empty leaf); split gain
// ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ. The ensemble score is
// F(x) = η·Σₜ treeₜ(x) (base score 0 ⇒ p = 0.5 before the first tree).
//
// Three growth policies share this engine:
//  • XGBoost  — level-wise: split every node down to max_depth.
//  • LightGBM — leaf-wise (best-first) up to num_leaves, and histogram binning:
//               each feature is cut once into ≤ max_bin equal-frequency bins and
//               split thresholds are restricted to those bin edges.
//  • CatBoost — symmetric (oblivious) trees: one shared test per level. In
//               "ordered" mode the tree STRUCTURE is chosen from ordered
//               gradients: sample σ(k) (k-th in a seeded random permutation σ)
//               gets its gradient from a supporting model M_k trained only on the
//               k samples before it, so no sample's own label leaks into its
//               gradient. Every M_j is refit with the new structure using only
//               its first j samples (CatBoost's Algorithm 2 with one permutation).
//               The final model's leaf values use all samples, as in CatBoost.

import { mulberry32, permutation } from './supData';

export type Variant = 'xgboost' | 'lightgbm' | 'catboost';
export type CatMode = 'ordered' | 'plain';
export interface GPt { x: number; y: number; y01: number; }

export type BNode =
  | { leaf: true; val: number; n: number }
  | { leaf: false; feat: 0 | 1; thr: number; gain: number; n: number; left: BNode; right: BNode };

export interface Grad { g: number; h: number; }

export interface BoostCfg {
  variant: Variant;
  lr: number;         // shrinkage η
  maxDepth: number;   // XGBoost / CatBoost depth
  numLeaves: number;  // LightGBM leaf budget
  maxBin: number;     // LightGBM histogram bins per feature (≥ #distinct ⇒ exact)
  lambda: number;     // L2 on leaf weights
  gamma: number;      // min split gain
  minLeaf: number;    // min samples per child
  catMode: CatMode;   // CatBoost boosting type
  permSeed: number;   // CatBoost permutation seed
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const logloss = (y: number, p: number) => -(y * Math.log(p + 1e-9) + (1 - y) * Math.log(1 - p + 1e-9));
const HMIN = 1e-6;
export const gradOf = (score: number, y: number): Grad => {
  const p = sigmoid(score);
  return { g: p - y, h: Math.max(HMIN, p * (1 - p)) };
};

const fv = (p: GPt, f: 0 | 1) => (f === 0 ? p.x : p.y);

/** Newton leaf weight −ΣG/(ΣH+λ); an empty leaf gets 0 (avoids 0/0 when λ = 0). */
export function leafWeight(idx: readonly number[], gr: readonly Grad[], lambda: number): number {
  if (idx.length === 0) return 0;
  let G = 0, H = 0;
  for (const i of idx) { G += gr[i]!.g; H += gr[i]!.h; }
  return -G / (H + lambda);
}

export function splitGain(L: readonly number[], R: readonly number[], gr: readonly Grad[], lambda: number, gamma: number): number {
  let GL = 0, HL = 0, GR = 0, HR = 0;
  for (const i of L) { GL += gr[i]!.g; HL += gr[i]!.h; }
  for (const i of R) { GR += gr[i]!.g; HR += gr[i]!.h; }
  const G = GL + GR, H = HL + HR;
  return 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) - G * G / (H + lambda)) - gamma;
}

/** Midpoints between consecutive distinct values: every exact split threshold. */
function exactThresholds(vals: number[]): number[] {
  const d = [...new Set(vals)].sort((a, b) => a - b);
  const out: number[] = [];
  for (let i = 0; i < d.length - 1; i++) out.push((d[i]! + d[i + 1]!) / 2);
  return out;
}

/**
 * Histogram bin edges of one feature: ≤ maxBin equal-frequency bins. With no
 * more distinct values than maxBin every value is its own bin (exact splits).
 */
export function binEdges(vals: readonly number[], maxBin: number): number[] {
  const d = [...new Set(vals)].sort((a, b) => a - b);
  if (d.length <= maxBin) return exactThresholds(d);
  const s = vals.slice().sort((a, b) => a - b), n = s.length;
  const edges: number[] = [];
  for (let k = 1; k < maxBin; k++) {
    const i = Math.round((k * n) / maxBin);
    const lo = s[i - 1], hi = s[i];
    if (lo === undefined || hi === undefined || hi <= lo) continue;
    const e = (lo + hi) / 2;
    if (!edges.length || e > edges[edges.length - 1]!) edges.push(e);
  }
  return edges;
}

interface Split { feat: 0 | 1; thr: number; L: number[]; R: number[]; gain: number; }
type Cands = [number[], number[]] | null; // per-feature thresholds (LightGBM bins) or null = exact

function bestSplit(pts: readonly GPt[], idx: readonly number[], gr: readonly Grad[], cfg: BoostCfg, cands: Cands): Split | null {
  let best: Split | null = null;
  for (const feat of [0, 1] as const) {
    const thrs = cands ? cands[feat] : exactThresholds(idx.map((i) => fv(pts[i]!, feat)));
    for (const thr of thrs) {
      const L: number[] = [], R: number[] = [];
      for (const i of idx) (fv(pts[i]!, feat) <= thr ? L : R).push(i);
      if (L.length < cfg.minLeaf || R.length < cfg.minLeaf) continue;
      const gain = splitGain(L, R, gr, cfg.lambda, cfg.gamma);
      if (!best || gain > best.gain) best = { feat, thr, L, R, gain };
    }
  }
  return best;
}

/** XGBoost: level-wise to max depth (a node splits only if its best gain > 0). */
function buildLevelWise(pts: readonly GPt[], idx: number[], gr: readonly Grad[], depth: number, cfg: BoostCfg): BNode {
  const leaf = (): BNode => ({ leaf: true, val: leafWeight(idx, gr, cfg.lambda), n: idx.length });
  if (depth >= cfg.maxDepth || idx.length < 2 * cfg.minLeaf) return leaf();
  const s = bestSplit(pts, idx, gr, cfg, null);
  if (!s || s.gain <= 0) return leaf();
  return { leaf: false, feat: s.feat, thr: s.thr, gain: s.gain, n: idx.length, left: buildLevelWise(pts, s.L, gr, depth + 1, cfg), right: buildLevelWise(pts, s.R, gr, depth + 1, cfg) };
}

/** LightGBM: best-first — repeatedly split the leaf with the largest positive gain. */
function buildLeafWise(pts: readonly GPt[], root: number[], gr: readonly Grad[], cfg: BoostCfg, cands: Cands): BNode {
  type Leaf = { leaf: true; val: number; n: number };
  interface Open { idx: number[]; split: Split | null; set: (n: BNode) => void; }
  const mk = (idx: number[]): Leaf => ({ leaf: true, val: leafWeight(idx, gr, cfg.lambda), n: idx.length });
  let rootRef: BNode = mk(root);
  const open: Open[] = [{ idx: root, split: bestSplit(pts, root, gr, cfg, cands), set: (n) => { rootRef = n; } }];
  for (let count = 1; count < cfg.numLeaves; count++) {
    let bi = -1, bg = 0;
    open.forEach((l, i) => { if (l.split && l.split.gain > bg) { bg = l.split.gain; bi = i; } });
    if (bi < 0) break;
    const o = open[bi]!, s = o.split!;
    const branch: BNode = { leaf: false, feat: s.feat, thr: s.thr, gain: s.gain, n: o.idx.length, left: mk(s.L), right: mk(s.R) };
    o.set(branch);
    open.splice(bi, 1);
    open.push({ idx: s.L, split: bestSplit(pts, s.L, gr, cfg, cands), set: (n) => { branch.left = n; } });
    open.push({ idx: s.R, split: bestSplit(pts, s.R, gr, cfg, cands), set: (n) => { branch.right = n; } });
  }
  return rootRef;
}

export interface ObliviousTest { feat: 0 | 1; thr: number; gain: number; }

/** CatBoost: choose one (feature, threshold) per level maximising the gain summed over all nodes. */
export function chooseObliviousTests(pts: readonly GPt[], gr: readonly Grad[], cfg: BoostCfg): ObliviousTest[] {
  const all = pts.map((_, i) => i);
  let groups: number[][] = [all];
  const tests: ObliviousTest[] = [];
  const thr: [number[], number[]] = [exactThresholds(pts.map((p) => p.x)), exactThresholds(pts.map((p) => p.y))];
  for (let d = 0; d < cfg.maxDepth; d++) {
    let best: ObliviousTest | null = null;
    for (const feat of [0, 1] as const) {
      for (const t of thr[feat]) {
        let total = 0, ok = false;
        for (const grp of groups) {
          const L: number[] = [], R: number[] = [];
          for (const i of grp) (fv(pts[i]!, feat) <= t ? L : R).push(i);
          if (L.length >= cfg.minLeaf && R.length >= cfg.minLeaf) { total += splitGain(L, R, gr, cfg.lambda, cfg.gamma); ok = true; }
        }
        if (ok && (!best || total > best.gain)) best = { feat, thr: t, gain: total };
      }
    }
    if (!best || best.gain <= 0) break;
    const b = best;
    tests.push(b);
    groups = groups.flatMap((grp) => {
      const L: number[] = [], R: number[] = [];
      for (const i of grp) (fv(pts[i]!, b.feat) <= b.thr ? L : R).push(i);
      return [L, R];
    });
  }
  return tests;
}

/** Leaf index (0 … 2^depth − 1) of a point in an oblivious tree: one bit per level. */
export const obliviousLeaf = (tests: readonly ObliviousTest[], x: number, y: number): number =>
  tests.reduce((k, t) => 2 * k + ((t.feat === 0 ? x : y) <= t.thr ? 0 : 1), 0);

/** Materialise an oblivious tree with the given per-leaf values. */
function obliviousTree(tests: readonly ObliviousTest[], vals: readonly number[], counts: readonly number[]): BNode {
  const build = (level: number, k: number): BNode => {
    if (level >= tests.length) return { leaf: true, val: vals[k] ?? 0, n: counts[k] ?? 0 };
    const t = tests[level]!;
    const left = build(level + 1, 2 * k), right = build(level + 1, 2 * k + 1);
    return { leaf: false, feat: t.feat, thr: t.thr, gain: t.gain, n: nOf(left) + nOf(right), left, right };
  };
  return build(0, 0);
}
const nOf = (n: BNode) => n.n;

export const predictTree = (n: BNode, x: number, y: number): number =>
  n.leaf ? n.val : predictTree((n.feat === 0 ? x : y) <= n.thr ? n.left : n.right, x, y);

/* ───────────────────────────── boosting state ───────────────────────────── */

export interface BoostState {
  trees: BNode[];            // leaf values are Newton weights (unscaled); F = η·Σ trees
  F: number[];               // final-model raw score at each training point
  bins: [number[], number[]] | null;   // LightGBM bin edges (x, y)
  perm: number[] | null;     // CatBoost σ: position → sample index
  pos: number[] | null;      // inverse of σ: sample index → position
  prefix: Float64Array[] | null;       // prefix[j][p] = M_j(x_{σ(p)}), p ≤ j
  lastTests: ObliviousTest[] | null;   // CatBoost: tests of the newest tree
}

export function initBoost(pts: readonly GPt[], cfg: BoostCfg): BoostState {
  const n = pts.length;
  const bins: [number[], number[]] | null = cfg.variant === 'lightgbm'
    ? [binEdges(pts.map((p) => p.x), cfg.maxBin), binEdges(pts.map((p) => p.y), cfg.maxBin)] : null;
  let perm: number[] | null = null, pos: number[] | null = null, prefix: Float64Array[] | null = null;
  if (cfg.variant === 'catboost' && cfg.catMode === 'ordered') {
    perm = permutation(n, mulberry32(cfg.permSeed));
    pos = new Array<number>(n).fill(0);
    perm.forEach((s, k) => { pos![s] = k; });
    prefix = Array.from({ length: n }, (_, j) => new Float64Array(j + 1));
  }
  return { trees: [], F: new Array<number>(n).fill(0), bins, perm, pos, prefix, lastTests: null };
}

/** Ordered gradients: sample σ(k) scored by M_k, the model fit on the k samples before it. */
export function orderedScores(st: BoostState, n: number): number[] {
  const s = new Array<number>(n).fill(0);
  if (!st.perm || !st.prefix) return s;
  for (let k = 0; k < n; k++) s[st.perm[k]!] = st.prefix[k]![k]!;
  return s;
}

/** Add one tree. Returns a new state (arrays copied, so React state stays immutable). */
export function boostRound(prev: BoostState, pts: readonly GPt[], cfg: BoostCfg): BoostState {
  const n = pts.length;
  const F = prev.F.slice();
  const plain: Grad[] = pts.map((p, i) => gradOf(F[i]!, p.y01));
  let tree: BNode;
  let prefix = prev.prefix;
  let lastTests: ObliviousTest[] | null = null;
  if (cfg.variant === 'catboost') {
    const ordered = cfg.catMode === 'ordered' && prev.perm && prev.prefix;
    const structGrads = ordered ? orderedScores(prev, n).map((s, i) => gradOf(s, pts[i]!.y01)) : plain;
    const tests = chooseObliviousTests(pts, structGrads, cfg);
    lastTests = tests;
    const L = 1 << tests.length;
    const leafOf = pts.map((p) => obliviousLeaf(tests, p.x, p.y));
    // Final model: Newton leaf values from ALL samples' plain gradients.
    const members: number[][] = Array.from({ length: L }, () => []);
    leafOf.forEach((k, i) => members[k]!.push(i));
    const vals = members.map((m) => leafWeight(m, plain, cfg.lambda));
    tree = obliviousTree(tests, vals, members.map((m) => m.length));
    // Supporting models: refit each M_j on its own first j samples.
    if (ordered && prev.perm && prev.prefix) {
      const perm = prev.perm;
      prefix = prev.prefix.map((row) => new Float64Array(row));
      const G = new Float64Array(L), H = new Float64Array(L);
      for (let j = 0; j < n; j++) {
        G.fill(0); H.fill(0);
        const row = prefix[j]!;
        for (let p = 0; p < j; p++) {
          const i = perm[p]!;
          const gh = gradOf(row[p]!, pts[i]!.y01);
          const k = leafOf[i]!;
          G[k] = G[k]! + gh.g; H[k] = H[k]! + gh.h;
        }
        for (let p = 0; p <= j; p++) {
          const k = leafOf[perm[p]!]!;
          const v = H[k]! > 0 ? -G[k]! / (H[k]! + cfg.lambda) : 0;
          row[p] = row[p]! + cfg.lr * v;
        }
      }
    }
  } else if (cfg.variant === 'lightgbm') {
    tree = buildLeafWise(pts, pts.map((_, i) => i), plain, cfg, prev.bins);
  } else {
    tree = buildLevelWise(pts, pts.map((_, i) => i), plain, 0, cfg);
  }
  for (let i = 0; i < n; i++) F[i] = F[i]! + cfg.lr * predictTree(tree, pts[i]!.x, pts[i]!.y);
  return { ...prev, trees: [...prev.trees, tree], F, prefix, lastTests };
}

export const scoreAt = (trees: readonly BNode[], lr: number, x: number, y: number) =>
  lr * trees.reduce((s, t) => s + predictTree(t, x, y), 0);

/** Mean log-loss and accuracy of raw scores against labels. */
export function lossAcc(scores: readonly number[], pts: readonly GPt[]): { loss: number; acc: number } {
  if (!pts.length) return { loss: 0, acc: 0 };
  let loss = 0, ok = 0;
  pts.forEach((p, i) => {
    const prob = sigmoid(scores[i]!);
    loss += logloss(p.y01, prob);
    if ((prob >= 0.5 ? 1 : 0) === p.y01) ok++;
  });
  return { loss: loss / pts.length, acc: ok / pts.length };
}

export function treeShape(n: BNode): { depth: number; leaves: number } {
  if (n.leaf) return { depth: 0, leaves: 1 };
  const l = treeShape(n.left), r = treeShape(n.right);
  return { depth: 1 + Math.max(l.depth, r.depth), leaves: l.leaves + r.leaves };
}
