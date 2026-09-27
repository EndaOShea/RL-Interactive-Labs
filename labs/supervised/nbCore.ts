// Naive Bayes for the Naive Bayes lab (pure maths, no React).
//
// Gaussian NB: per class c and feature f, the MLE mean μ_{c,f} and variance
// σ²_{c,f} = mean squared deviation (floored at VAR_FLOOR). Multinomial NB: each
// axis is cut into NB_BINS equal cells on [0,1]; the per-class cell counts give
// P(cell | c) = (count + α) / (N_c + α·B) per axis (Laplace smoothing). Both use
// log-space posteriors log P(c) + Σ_f log P(x_f | c); the prior is the class
// frequency n_c / n ("learned") or 1/K ("uniform").

export const NB_BINS = 8;
export const VAR_FLOOR = 1e-3;
export type NbVariant = 'gaussian' | 'multinomial';

export interface NbPoint { x: number; y: number; cls: number; }
export interface NbClassStats { n: number; mx: number; my: number; vx: number; vy: number; }
export interface NbModel { K: number; n: number; s: NbClassStats[]; hx: number[][]; hy: number[][]; pooled: { vx: number; vy: number }; }

export const binOf = (v: number) => Math.min(NB_BINS - 1, Math.max(0, Math.floor(v * NB_BINS)));

export function fitNb(data: readonly NbPoint[], K: number): NbModel {
  const s: NbClassStats[] = Array.from({ length: K }, () => ({ n: 0, mx: 0, my: 0, vx: 0, vy: 0 }));
  for (const p of data) { const c = s[p.cls]!; c.n++; c.mx += p.x; c.my += p.y; }
  for (const c of s) if (c.n) { c.mx /= c.n; c.my /= c.n; }
  for (const p of data) { const c = s[p.cls]!; c.vx += (p.x - c.mx) ** 2; c.vy += (p.y - c.my) ** 2; }
  // Pooled (shared) within-class variance — only used to measure how curved the per-class boundary is.
  let px = 0, py = 0;
  for (const c of s) { px += c.vx; py += c.vy; }
  const pooled = { vx: Math.max(VAR_FLOOR, px / Math.max(1, data.length)), vy: Math.max(VAR_FLOOR, py / Math.max(1, data.length)) };
  for (const c of s) { c.vx = Math.max(VAR_FLOOR, c.vx / Math.max(1, c.n)); c.vy = Math.max(VAR_FLOOR, c.vy / Math.max(1, c.n)); }
  const hx = Array.from({ length: K }, () => new Array<number>(NB_BINS).fill(0));
  const hy = Array.from({ length: K }, () => new Array<number>(NB_BINS).fill(0));
  for (const p of data) {
    const rx = hx[p.cls]!, ry = hy[p.cls]!, bx = binOf(p.x), by = binOf(p.y);
    rx[bx] = (rx[bx] ?? 0) + 1; ry[by] = (ry[by] ?? 0) + 1;
  }
  return { K, n: data.length, s, hx, hy, pooled };
}

export interface NbOpts { variant: NbVariant; alpha: number; learnedPrior: boolean; pooledVar?: boolean; }

export const logPrior = (m: NbModel, c: number, learned: boolean) => {
  const k = m.s[c]!;
  if (!k.n) return -Infinity;
  return learned ? Math.log(k.n / m.n) : Math.log(1 / m.K);
};

const logGauss = (v: number, mu: number, s2: number) => -0.5 * Math.log(2 * Math.PI * s2) - (v - mu) ** 2 / (2 * s2);

/** log P(x | c): the naive product of per-feature likelihoods, in log space. */
export function logLik(m: NbModel, o: NbOpts, x: number, y: number, c: number): number {
  const k = m.s[c]!;
  if (o.variant === 'gaussian') {
    const vx = o.pooledVar ? m.pooled.vx : k.vx, vy = o.pooledVar ? m.pooled.vy : k.vy;
    return logGauss(x, k.mx, vx) + logGauss(y, k.my, vy);
  }
  const hx = m.hx[c]!, hy = m.hy[c]!;
  const px = (hx[binOf(x)]! + o.alpha) / (k.n + o.alpha * NB_BINS);
  const py = (hy[binOf(y)]! + o.alpha) / (k.n + o.alpha * NB_BINS);
  return Math.log(px) + Math.log(py);
}

export const logPost = (m: NbModel, o: NbOpts, x: number, y: number, c: number) =>
  logPrior(m, c, o.learnedPrior) + logLik(m, o, x, y, c);

export function posteriors(m: NbModel, o: NbOpts, x: number, y: number): number[] {
  const lp = m.s.map((_, c) => logPost(m, o, x, y, c));
  const mx = Math.max(...lp);
  const ex = lp.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((v) => v / sum);
}

/** argmax_c log P(c | x) — ties go to the lower class index. */
export function predictNb(m: NbModel, o: NbOpts, x: number, y: number): number {
  let best = 0, bv = -Infinity;
  for (let c = 0; c < m.K; c++) { const v = logPost(m, o, x, y, c); if (v > bv) { bv = v; best = c; } }
  return best;
}

export function accuracyNb(m: NbModel, o: NbOpts, data: readonly NbPoint[]): number {
  if (!data.length) return 0;
  let ok = 0;
  for (const p of data) if (predictNb(m, o, p.x, p.y) === p.cls) ok++;
  return ok / data.length;
}

/** Fraction of [0,1]² (an M×M lattice of cell centres) where two decision rules disagree. */
export function disagreement(a: (x: number, y: number) => number, b: (x: number, y: number) => number, M = 80): number {
  let d = 0;
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
    const x = (i + 0.5) / M, y = (j + 0.5) / M;
    if (a(x, y) !== b(x, y)) d++;
  }
  return d / (M * M);
}
