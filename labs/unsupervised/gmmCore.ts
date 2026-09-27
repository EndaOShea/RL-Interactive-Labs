// Gaussian-mixture EM for the GMM lab. Pure maths, no React.
//
// Conventions (mirrored by the Python export):
//  • init   — means = k-means++ seeds drawn from the data, Σ₀ = INIT_VAR·I, π₀ = 1/K
//  • E-step — responsibilities in log space (log-sum-exp), so a point far from
//             every component still gets exact posteriors and a finite logL
//  • M-step — weighted means / covariances, then the covariance family's
//             constraint, then + REG_COVAR on the diagonal (scikit-learn's
//             reg_covar), which keeps every Σ positive definite
//  • a component whose total responsibility N_k falls below STARVED keeps its
//    previous μ and Σ (dividing by ~0 would throw it to the origin)
//  • stop when |Δ logL| < TOL (total log-likelihood)
import type { UPt } from './shared';
import { logGauss2 } from './shared';
import type { Rng } from '../classic-ml/rng';
import { kppSeeds } from '../classic-ml/kmeansCore';

export type Cov = [number, number, number]; // [a, b, c] for [[a,b],[b,c]]
export type CovType = 'full' | 'diag' | 'spherical';
export interface GmmParams { means: UPt[]; covs: Cov[]; weights: number[] }

export const INIT_VAR = 0.02;
export const REG_COVAR = 1e-4;
export const STARVED = 1e-8;
export const TOL = 1e-4;

/** Free parameters per component, means included: full 2+3, diag 2+2, spherical 2+1. */
export const PARAMS_PER_COMP: Record<CovType, number> = { full: 5, diag: 4, spherical: 3 };
/** Total free parameters: K−1 mixing weights + K·(mean + covariance) — as scikit-learn counts them. */
export const nParams = (K: number, covType: CovType) => (K - 1) + K * PARAMS_PER_COMP[covType];
export const bicOf = (logLik: number, K: number, covType: CovType, n: number) => -2 * logLik + nParams(K, covType) * Math.log(n);

export function initGmm(pts: UPt[], K: number, r: Rng): GmmParams {
  const idx = kppSeeds(pts, K, r);
  return {
    means: idx.map((i) => ({ x: pts[i]!.x, y: pts[i]!.y })),
    covs: Array.from({ length: K }, () => [INIT_VAR, 0, INIT_VAR] as Cov),
    weights: Array.from({ length: K }, () => 1 / K),
  };
}

/** E-step: γ_ik = π_k 𝒩(x_i|μ_k,Σ_k) / Σ_j π_j 𝒩(x_i|μ_j,Σ_j), computed with log-sum-exp. */
export function eStep(pts: UPt[], P: GmmParams) {
  const K = P.means.length;
  const resp: number[][] = [];
  let logLik = 0;
  const lw = P.weights.map((w) => Math.log(w));
  for (const p of pts) {
    const l = new Array<number>(K);
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      const mu = P.means[k]!, S = P.covs[k]!;
      l[k] = lw[k]! + logGauss2(p.x, p.y, mu.x, mu.y, S[0], S[1], S[2]);
      if (l[k]! > m) m = l[k]!;
    }
    let s = 0;
    for (let k = 0; k < K; k++) s += Math.exp(l[k]! - m);
    const lse = m + Math.log(s);
    logLik += lse;
    resp.push(l.map((v) => Math.exp(v - lse)));
  }
  return { resp, logLik };
}

/** Constrain a fitted covariance to the family, then add the regulariser to the diagonal. */
export function constrainCov(a: number, b: number, c: number, type: CovType): Cov {
  if (type === 'spherical') { const s = (a + c) / 2 + REG_COVAR; return [s, 0, s]; }
  if (type === 'diag') return [a + REG_COVAR, 0, c + REG_COVAR];
  return [a + REG_COVAR, b, c + REG_COVAR];
}

/** M-step: refit every component to its responsibility-weighted points. */
export function mStep(pts: UPt[], resp: number[][], covType: CovType, prev: GmmParams) {
  const K = prev.means.length, n = pts.length;
  const means: UPt[] = [], covs: Cov[] = [], weights: number[] = [];
  const starved: number[] = [];
  for (let k = 0; k < K; k++) {
    let Nk = 0, mx = 0, my = 0;
    for (let i = 0; i < n; i++) { const r = resp[i]![k]!; Nk += r; mx += r * pts[i]!.x; my += r * pts[i]!.y; }
    weights.push(Nk / n);
    if (Nk < STARVED) {
      starved.push(k);
      means.push({ ...prev.means[k]! });
      covs.push([...prev.covs[k]!] as Cov);
      continue;
    }
    mx /= Nk; my /= Nk;
    let a = 0, b = 0, c = 0;
    for (let i = 0; i < n; i++) {
      const r = resp[i]![k]!;
      const dx = pts[i]!.x - mx, dy = pts[i]!.y - my;
      a += r * dx * dx; b += r * dx * dy; c += r * dy * dy;
    }
    means.push({ x: mx, y: my });
    covs.push(constrainCov(a / Nk, b / Nk, c / Nk, covType));
  }
  return { params: { means, covs, weights } as GmmParams, starved };
}

/** Fit to convergence (harness / verification helper; the lab steps the same functions one at a time). */
export function fitGmm(pts: UPt[], init: GmmParams, covType: CovType, maxIter = 500) {
  let P = init;
  let prev = -Infinity;
  let iters = 0;
  for (; iters < maxIter; iters++) {
    const e = eStep(pts, P);
    P = mStep(pts, e.resp, covType, P).params;
    const gain = e.logLik - prev;
    prev = e.logLik;
    if (Math.abs(gain) < TOL) { iters++; break; }
  }
  const final = eStep(pts, P);
  return { params: P, logLik: final.logLik, resp: final.resp, iters, bic: bicOf(final.logLik, P.means.length, covType, pts.length) };
}
