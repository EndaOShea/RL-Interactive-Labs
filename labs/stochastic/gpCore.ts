// Pure maths for the Gaussian-process lab (no React): the baked data set, the
// closed-form posterior via a Cholesky factorisation (no explicit inverse), the
// log marginal likelihood, and posterior / prior sample functions.
import { rng, gaussFrom, kernel, gram, cross, cholesky, forwardSub, cholSolve, dot } from './shared';
import type { KernelId, Vec, Mat } from './shared';

export const NXS = 140;          // fine grid for mean / band
export const NXC = 56;           // coarse grid for sampled functions
export const N_SAMP = 4;         // sample functions drawn
export const PERIOD = 0.3;       // period of the periodic kernel
export const DATA_SEED = 7;
export const DATA_NOISE = 0.06;  // std of the noise baked into the 8 observations
export const SAMPLE_JITTER = 1e-6;

export const fTrue = (x: number) => Math.sin(2 * Math.PI * x) * 0.7;

/** Fixed data with a gap, revealed one point at a time during Run. */
export const DATA_SET = (() => {
  const r = rng(DATA_SEED);
  const xs = [0.07, 0.16, 0.25, 0.33, 0.60, 0.70, 0.82, 0.93];
  const ys = xs.map((x) => fTrue(x) + gaussFrom(r) * DATA_NOISE);
  return { xs, ys };
})();
export const XS = Array.from({ length: NXS }, (_, i) => i / (NXS - 1));
export const XC = Array.from({ length: NXC }, (_, i) => i / (NXC - 1));
export const IDX_GAP = Math.round(0.46 * (NXS - 1));    // XS[64] ≈ 0.460, inside the gap
export const IDX_DATA = Math.round(0.16 * (NXS - 1));   // XS[22] ≈ 0.158, beside the point at 0.16

export interface GpPosterior {
  mean: number[];           // posterior mean on XS
  std: number[];            // posterior std of f on XS (latent function, no noise)
  logML: number | null;     // log p(y | X, θ) — null before any data
}

/**
 * Posterior of f on the fine grid given the first observations:
 * μ∗ = k∗ᵀ(K+σ²ₙI)⁻¹y and σ²∗ = k(x,x) − k∗ᵀ(K+σ²ₙI)⁻¹k∗, both through the
 * Cholesky factor L of K+σ²ₙI (solves, never an inverse). Also returns the log
 * marginal likelihood −½yᵀ(K+σ²ₙI)⁻¹y − Σ log L_ii − (n/2) log 2π.
 */
export function gpPosterior(kid: KernelId, ell: number, sf: number, sn: number, obsX: Vec, obsY: Vec): GpPosterior {
  const n = obsX.length;
  const mean = new Array(NXS).fill(0);
  const std = new Array(NXS).fill(0);
  if (n === 0) {
    for (let g = 0; g < NXS; g++) std[g] = Math.sqrt(Math.max(0, kernel(kid, XS[g] ?? 0, XS[g] ?? 0, ell, sf, PERIOD)));
    return { mean, std, logML: null };
  }
  const K = gram(kid, obsX, ell, sf, PERIOD).map((row, i) => row.map((v, j) => v + (i === j ? sn * sn : 0)));
  const L = cholesky(K, 0);
  const a = cholSolve(L, obsY);
  for (let g = 0; g < NXS; g++) {
    const xg = XS[g] ?? 0;
    const ks = obsX.map((xo) => kernel(kid, xg, xo, ell, sf, PERIOD));
    mean[g] = dot(ks, a);
    const v = forwardSub(L, ks);
    std[g] = Math.sqrt(Math.max(0, kernel(kid, xg, xg, ell, sf, PERIOD) - dot(v, v)));
  }
  let logDetHalf = 0;
  for (let i = 0; i < n; i++) logDetHalf += Math.log(L[i]?.[i] ?? 1);
  const logML = -0.5 * dot(obsY, a) - logDetHalf - (n / 2) * Math.log(2 * Math.PI);
  return { mean, std, logML };
}

/** The lab's slider grids — hyperparameter fitting searches exactly these values. */
export const ELL_RANGE = { min: 0.03, max: 0.6, step: 0.005 };
export const SF_RANGE = { min: 0.3, max: 2, step: 0.05 };
export const SN_RANGE = { min: 0.01, max: 0.3, step: 0.005 };
const gridOf = (r: { min: number; max: number; step: number }) =>
  Array.from({ length: Math.round((r.max - r.min) / r.step) + 1 }, (_, i) => +(r.min + i * r.step).toFixed(4));

export interface HyperFit { ell: number; sf: number; sn: number; logML: number; sweeps: number; }

/**
 * Type-II maximum likelihood: coordinate ascent of the log marginal likelihood
 * over the slider grids (ℓ, then σ_f, then σ_n, repeated until no value improves
 * it). Starts from the current setting, so it only ever moves uphill.
 */
export function fitHyper(kid: KernelId, obsX: Vec, obsY: Vec, start: { ell: number; sf: number; sn: number }): HyperFit {
  const lml = (l: number, f: number, s: number) => gpPosterior(kid, l, f, s, obsX, obsY).logML ?? -Infinity;
  let { ell, sf, sn } = start;
  let best = lml(ell, sf, sn);
  let sweeps = 0;
  const ells = gridOf(ELL_RANGE), sfs = gridOf(SF_RANGE), sns = gridOf(SN_RANGE);
  for (; sweeps < 30; sweeps++) {
    const before = best;
    for (const l of ells) { const v = lml(l, sf, sn); if (v > best) { best = v; ell = l; } }
    for (const f of sfs) { const v = lml(ell, f, sn); if (v > best) { best = v; sf = f; } }
    for (const s of sns) { const v = lml(ell, sf, s); if (v > best) { best = v; sn = s; } }
    if (best - before < 1e-9) { sweeps++; break; }
  }
  return { ell, sf, sn, logML: best, sweeps };
}

/** Sample functions on the coarse grid — prior when there is no data, else posterior. */
export function gpSamples(kid: KernelId, ell: number, sf: number, sn: number, obsX: Vec, obsY: Vec, rand: () => number): number[][] {
  const Kss = gram(kid, XC, ell, sf, PERIOD);
  let meanC: Vec;
  let covC: Mat;
  if (obsX.length === 0) {
    meanC = new Array(NXC).fill(0);
    covC = Kss;
  } else {
    const K = gram(kid, obsX, ell, sf, PERIOD).map((row, i) => row.map((v, j) => v + (i === j ? sn * sn : 0)));
    const L = cholesky(K, 0);
    const Ksc = cross(kid, XC, obsX, ell, sf, PERIOD);            // NXC × n
    const a = cholSolve(L, obsY);
    meanC = Ksc.map((ks) => dot(ks, a));
    const V = Ksc.map((ks) => forwardSub(L, ks));                 // row c = L⁻¹k∗(c)
    covC = Kss.map((row, i) => row.map((v, j) => v - dot(V[i] ?? [], V[j] ?? [])));
  }
  const Lc = cholesky(covC, SAMPLE_JITTER);
  return Array.from({ length: N_SAMP }, () => {
    const z = meanC.map(() => gaussFrom(rand));
    return meanC.map((mv, r) => mv + dot(Lc[r] ?? [], z));
  });
}
