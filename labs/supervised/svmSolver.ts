// Soft-margin SVM trained by SMO — the same algorithm LIBSVM (and so scikit-learn's
// SVC) uses. It solves the dual of  min ½‖w‖² + C·Σᵢ max(0, 1 − yᵢ f(xᵢ)):
//
//     min_α  ½ αᵀQα − Σᵢ αᵢ     s.t.  0 ≤ αᵢ ≤ C,  Σᵢ yᵢαᵢ = 0,   Qᵢⱼ = yᵢyⱼK(xᵢ,xⱼ)
//
// Each SMO iteration picks the maximal-violating pair (i, j) with LIBSVM's
// second-order working-set selection (WSS3, Fan–Chen–Lin 2005), solves the
// two-variable sub-problem in closed form, clips it to the box, and updates the
// gradient G = Qα − 1. It stops when the KKT gap m(α) − M(α) < tol (sklearn's
// default tol = 1e-3). The bias follows LIBSVM's calculate_rho. Pure module.
import type { SvmPt } from './supData';

export type Kernel = 'linear' | 'poly' | 'rbf';
export interface KernelCfg { kernel: Kernel; gamma: number; degree: number; }

export const SMO_TOL = 1e-3;
const TAU = 1e-12;

/** K(a, b). Poly uses repeated multiplication so the Python port matches bit-for-bit. */
export function kernelValue(cfg: KernelCfg, ax: number, ay: number, bx: number, by: number): number {
  if (cfg.kernel === 'linear') return ax * bx + ay * by;
  if (cfg.kernel === 'poly') {
    const base = 1 + ax * bx + ay * by;
    let p = 1;
    for (let k = 0; k < cfg.degree; k++) p *= base;
    return p;
  }
  const dx = ax - bx, dy = ay - by;
  return Math.exp(-cfg.gamma * (dx * dx + dy * dy));
}

/** Dense n×n kernel matrix (row-major). n ≤ 140 here, so 20k entries at most. */
export function kernelMatrix(pts: readonly SvmPt[], cfg: KernelCfg): Float64Array {
  const n = pts.length;
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const a = pts[i]!;
    for (let j = i; j < n; j++) {
      const b = pts[j]!;
      const v = kernelValue(cfg, a.x, a.y, b.x, b.y);
      K[i * n + j] = v; K[j * n + i] = v;
    }
  }
  return K;
}

export interface SmoState {
  alpha: Float64Array;   // dual coefficients, 0 ≤ αᵢ ≤ C
  G: Float64Array;       // gradient of the dual objective, G = Qα − 1
  iter: number;          // pair updates performed
  gap: number;           // KKT violation m(α) − M(α) at the current α
  lastPair: [number, number] | null;
  done: boolean;         // gap < tol (or no admissible pair left)
}

export function initSmo(n: number): SmoState {
  return { alpha: new Float64Array(n), G: new Float64Array(n).fill(-1), iter: 0, gap: Infinity, lastPair: null, done: n === 0 };
}

const upper = (a: number, C: number) => a >= C;
const lower = (a: number) => a <= 0;

/** LIBSVM select_working_set (WSS3). Returns the pair and the KKT gap m − M. */
export function selectPair(st: SmoState, y: readonly number[], K: Float64Array, C: number): { i: number; j: number; gap: number } {
  const n = y.length;
  const { alpha, G } = st;
  let Gmax = -Infinity, Gmax2 = -Infinity, iIdx = -1, jIdx = -1, objMin = Infinity;
  for (let t = 0; t < n; t++) {
    const at = alpha[t]!, gt = G[t]!;
    if (y[t] === 1) { if (!upper(at, C) && -gt >= Gmax) { Gmax = -gt; iIdx = t; } }
    else if (!lower(at) && gt >= Gmax) { Gmax = gt; iIdx = t; }
  }
  if (iIdx < 0) return { i: -1, j: -1, gap: 0 };
  const Kii = K[iIdx * n + iIdx]!;
  for (let j = 0; j < n; j++) {
    const aj = alpha[j]!, gj = G[j]!;
    const quad0 = Kii + K[j * n + j]! - 2 * K[iIdx * n + j]!;
    if (y[j] === 1) {
      if (!lower(aj)) {
        const gd = Gmax + gj;
        if (gj >= Gmax2) Gmax2 = gj;
        if (gd > 0) {
          const obj = -(gd * gd) / (quad0 > 0 ? quad0 : TAU);
          if (obj <= objMin) { jIdx = j; objMin = obj; }
        }
      }
    } else if (!upper(aj, C)) {
      const gd = Gmax - gj;
      if (-gj >= Gmax2) Gmax2 = -gj;
      if (gd > 0) {
        const obj = -(gd * gd) / (quad0 > 0 ? quad0 : TAU);
        if (obj <= objMin) { jIdx = j; objMin = obj; }
      }
    }
  }
  return { i: iIdx, j: jIdx, gap: Gmax + Gmax2 };
}

/**
 * Run up to `maxSteps` SMO pair updates on a copy of `prev` (so React state stays
 * immutable). Stops early once the KKT gap drops below `tol`.
 */
export function smoRun(prev: SmoState, y: readonly number[], K: Float64Array, C: number, maxSteps: number, tol = SMO_TOL): SmoState {
  const n = y.length;
  const alpha = new Float64Array(prev.alpha), G = new Float64Array(prev.G);
  const st: SmoState = { alpha, G, iter: prev.iter, gap: prev.gap, lastPair: prev.lastPair, done: prev.done };
  for (let s = 0; s < maxSteps && !st.done; s++) {
    const { i, j, gap } = selectPair(st, y, K, C);
    st.gap = gap;
    if (i < 0 || j < 0 || gap < tol) { st.done = true; break; }
    const oi = alpha[i]!, oj = alpha[j]!;
    const yi = y[i]!, yj = y[j]!;
    let quad = K[i * n + i]! + K[j * n + j]! - 2 * K[i * n + j]!;
    if (quad <= 0) quad = TAU;
    let ai = oi, aj = oj;
    if (yi !== yj) {
      const delta = (-G[i]! - G[j]!) / quad;
      const diff = oi - oj;
      ai += delta; aj += delta;
      if (diff > 0) { if (aj < 0) { aj = 0; ai = diff; } }
      else if (ai < 0) { ai = 0; aj = -diff; }
      if (diff > 0) { if (ai > C) { ai = C; aj = C - diff; } }
      else if (aj > C) { aj = C; ai = C + diff; }
    } else {
      const delta = (G[i]! - G[j]!) / quad;
      const sum = oi + oj;
      ai -= delta; aj += delta;
      if (sum > C) { if (ai > C) { ai = C; aj = sum - C; } }
      else if (aj < 0) { aj = 0; ai = sum; }
      if (sum > C) { if (aj > C) { aj = C; ai = sum - C; } }
      else if (ai < 0) { ai = 0; aj = sum; }
    }
    alpha[i] = ai; alpha[j] = aj;
    const dai = ai - oi, daj = aj - oj;
    for (let k = 0; k < n; k++) {
      G[k] = G[k]! + yi * y[k]! * K[i * n + k]! * dai + yj * y[k]! * K[j * n + k]! * daj;
    }
    st.iter++;
    st.lastPair = [i, j];
  }
  if (!st.done) {
    const { i, j, gap } = selectPair(st, y, K, C);
    st.gap = gap;
    if (i < 0 || j < 0 || gap < tol) st.done = true;
  }
  return st;
}

/** LIBSVM calculate_rho: b = −ρ, ρ = mean of yᵢGᵢ over free SVs (0 < αᵢ < C). */
export function computeBias(st: SmoState, y: readonly number[], C: number): number {
  let ub = Infinity, lb = -Infinity, sumFree = 0, nFree = 0;
  for (let i = 0; i < y.length; i++) {
    const yG = y[i]! * st.G[i]!;
    const a = st.alpha[i]!;
    if (upper(a, C)) { if (y[i] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else if (lower(a)) { if (y[i] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else { nFree++; sumFree += yG; }
  }
  const rho = nFree > 0 ? sumFree / nFree : (Number.isFinite(ub + lb) ? (ub + lb) / 2 : 0);
  return -rho;
}

export interface SvmSummary {
  b: number;
  w: [number, number] | null;      // linear kernel only: w = Σ αᵢyᵢxᵢ
  sv: number;                      // αᵢ > 0
  freeSv: number;                  // 0 < αᵢ < C — on the margin, yᵢf(xᵢ) = 1
  boundSv: number;                 // αᵢ = C — inside the street or misclassified
  inside: number;                  // training points with yᵢf(xᵢ) < 1 (margin violations)
  wrong: number;                   // training points with yᵢf(xᵢ) ≤ 0
  dual: number;                    // Σα − ½αᵀQα (maximised)
  primal: number;                  // ½‖w‖² + C·Σ hinge at the current (α, b)
  sumAlpha: number;
}

/** Everything the lab displays about the current α (bias, SV counts, objectives). */
export function summarise(st: SmoState, pts: readonly SvmPt[], cfg: KernelCfg, C: number): SvmSummary {
  const y = pts.map((p) => p.yy);
  const b = computeBias(st, y, C);
  let sv = 0, freeSv = 0, boundSv = 0, inside = 0, wrong = 0, sumA = 0, quad = 0, hinge = 0;
  let w1 = 0, w2 = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = st.alpha[i]!, g = st.G[i]!, p = pts[i]!;
    sumA += a;
    quad += a * (g + 1);                      // αᵀQα = Σ αᵢ(Gᵢ + 1)
    if (a > 0) { sv++; if (a >= C) boundSv++; else freeSv++; }
    const m = g + 1 + p.yy * b;               // yᵢ f(xᵢ) = (Qα)ᵢ + yᵢb
    if (m < 1) { inside++; hinge += 1 - m; }
    if (m <= 0) wrong++;
    w1 += a * p.yy * p.x; w2 += a * p.yy * p.y;
  }
  return {
    b, w: cfg.kernel === 'linear' ? [w1, w2] : null, sv, freeSv, boundSv, inside, wrong,
    dual: sumA - 0.5 * quad, primal: 0.5 * quad + C * hinge, sumAlpha: sumA,
  };
}

/** f(x) = Σᵢ αᵢyᵢK(xᵢ, x) + b over the support vectors. */
export function decisionValue(x: number, y: number, pts: readonly SvmPt[], alpha: Float64Array, b: number, cfg: KernelCfg): number {
  let s = b;
  for (let i = 0; i < pts.length; i++) {
    const a = alpha[i]!;
    if (a === 0) continue;
    const p = pts[i]!;
    s += a * p.yy * kernelValue(cfg, p.x, p.y, x, y);
  }
  return s;
}

export interface Segment { x1: number; y1: number; x2: number; y2: number; }

/**
 * Marching squares: the level-set f = level of a function sampled on an
 * (n+1)×(n+1) lattice over [lo, hi]², as line segments with linear interpolation
 * along each cell edge. Saddle cells are resolved by the cell-centre average.
 */
export function contour(f: (x: number, y: number) => number, lo: number, hi: number, n: number, levels: readonly number[]): Segment[][] {
  const step = (hi - lo) / n;
  const vals = new Float64Array((n + 1) * (n + 1));
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) vals[j * (n + 1) + i] = f(lo + i * step, lo + j * step);
  return levels.map((lv) => {
    const segs: Segment[] = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v00 = vals[j * (n + 1) + i]! - lv, v10 = vals[j * (n + 1) + i + 1]! - lv;
        const v01 = vals[(j + 1) * (n + 1) + i]! - lv, v11 = vals[(j + 1) * (n + 1) + i + 1]! - lv;
        const x0 = lo + i * step, y0 = lo + j * step, x1 = x0 + step, y1 = y0 + step;
        // Crossing points on the four edges (bottom, right, top, left).
        const cross: { x: number; y: number }[] = [];
        const edge = (va: number, vb: number, ax: number, ay: number, bx: number, by: number) => {
          if ((va < 0) !== (vb < 0)) { const t = va / (va - vb); cross.push({ x: ax + t * (bx - ax), y: ay + t * (by - ay) }); }
          else cross.push({ x: NaN, y: NaN });
        };
        edge(v00, v10, x0, y0, x1, y0);   // bottom
        edge(v10, v11, x1, y0, x1, y1);   // right
        edge(v01, v11, x0, y1, x1, y1);   // top
        edge(v00, v01, x0, y0, x0, y1);   // left
        const hit = cross.map((c, k) => (Number.isNaN(c.x) ? -1 : k)).filter((k) => k >= 0);
        const at = (k: number) => cross[k]!;
        const seg = (a: number, b: number) => segs.push({ x1: at(a).x, y1: at(a).y, x2: at(b).x, y2: at(b).y });
        if (hit.length === 2) seg(hit[0]!, hit[1]!);
        else if (hit.length === 4) {
          const centre = (v00 + v10 + v01 + v11) / 4;
          // Pair each edge with the neighbour that keeps the centre's sign region connected.
          if ((centre < 0) === (v00 < 0)) { seg(0, 1); seg(2, 3); } else { seg(0, 3); seg(1, 2); }
        }
      }
    }
    return segs;
  });
}
