// Pure maths for the Bayesian-NN lab (no React). The "network" is a fixed random
// tanh feature layer φ(x) = tanh(w1·x + b1) with a Bayesian linear output layer,
// so the exact posterior over the output weights is Gaussian and closed-form.
// Each inference mode is a genuine, computed approximation of that model:
//   • point        — the single posterior-mean weight vector (no spread)
//   • exact        — samples w ~ N(m, Σ) of the exact full-covariance posterior
//   • variational  — mean-field VI q(w) = Π N(μ_i, s_i²), fitted by coordinate
//                    ascent (CAVI) on the ELBO; its optimum is the fixed point
//                    μ = m, s_i² = 1/A_ii (A = αI + βΦᵀΦ)
//   • dropout      — output layer fitted under the dropout-regularised objective,
//                    then the same Bernoulli masks applied at test time
//   • ensemble     — K networks with independent random feature layers, each
//                    fitted (posterior mean) on the same data
import { rng, gaussFrom, makeFeatures, features, bayesLinear, cholesky, matVec, transpose, matMul, invert, dot } from './shared';
import type { Vec, Mat } from './shared';

export const M = 24;                 // random tanh features
export const NXS = 161;              // plotting grid x = i/160, so 0.2 and 0.5 are grid points
export const ENS_K = 8;              // ensemble members
export const ENSEMBLE_SEED0 = 100;   // member k uses feature seed 100 + k
export const FEATURE_SEED = 7;       // the single network's feature seed
export const DATA_SEED = 20250608;
export const DROPOUT_P = 0.2;
export const IDX_GAP = 80;           // XS[80] = 0.5 — inside the data gap
export const IDX_DATA = 32;          // XS[32] = 0.2 — inside the left cluster
export const TARGET_SAMPLES = 28;    // curves drawn by the sampling modes

export type BnnMode = 'point' | 'exact' | 'variational' | 'dropout' | 'ensemble';

export const fTrue = (x: number) => 0.8 * Math.sin(2 * Math.PI * 1.3 * x);

/** Deterministic training data with a GAP in the middle and room to extrapolate. */
export const DATA_SET = (() => {
  const r = rng(DATA_SEED);
  const xs: number[] = [];
  for (let i = 0; i < 7; i++) xs.push(0.05 + (i / 6) * 0.30);   // left cluster
  for (let i = 0; i < 7; i++) xs.push(0.62 + (i / 6) * 0.33);   // right cluster
  const ys = xs.map((x) => fTrue(x) + gaussFrom(r) * 0.05);
  return { xs, ys };
})();
export const XS = Array.from({ length: NXS }, (_, i) => i / (NXS - 1));

const diagOf = (A: Mat): Vec => A.map((row, i) => row[i] ?? 0);

/** log|A| of a symmetric positive-definite matrix via its Cholesky factor. */
export function logDet(A: Mat): number {
  const L = cholesky(A, 0);
  let s = 0;
  for (let i = 0; i < L.length; i++) s += Math.log(L[i]?.[i] ?? 1);
  return 2 * s;
}

export interface MeanField {
  mu: Vec;          // variational means (= the exact posterior mean m)
  s2: Vec;          // variational variances 1/A_ii
  elbo: number;     // ELBO at the optimum (nats)
  logEvidence: number;
  kl: number;       // KL(q ‖ exact posterior) = log evidence − ELBO
  /** s_i² / Σ_ii across the weights — how much mean-field shrinks each weight's variance */
  varRatio: { min: number; median: number; max: number };
}

/**
 * ELBO of a factorised Gaussian q(w) = Π N(μ_i, s_i²) for Bayesian linear
 * regression y = Φw + ε, ε ~ N(0, 1/β), w ~ N(0, α⁻¹I):
 * E_q[ln p(y|w)] + E_q[ln p(w)] + H[q].
 */
export function elbo(Phi: Mat, y: Vec, alpha: number, beta: number, mu: Vec, s2: Vec): number {
  const n = y.length, m = mu.length;
  let rss = 0;
  for (let k = 0; k < n; k++) { const r = (y[k] ?? 0) - dot(Phi[k] ?? [], mu); rss += r * r; }
  let trace = 0;   // Σ_i s_i² (ΦᵀΦ)_ii
  for (let i = 0; i < m; i++) { let c = 0; for (let k = 0; k < n; k++) c += (Phi[k]?.[i] ?? 0) ** 2; trace += (s2[i] ?? 0) * c; }
  let mm = 0, ss = 0, ent = 0;
  for (let i = 0; i < m; i++) { mm += (mu[i] ?? 0) ** 2; ss += s2[i] ?? 0; ent += 0.5 * Math.log(2 * Math.PI * Math.E * (s2[i] ?? 1)); }
  const lik = (n / 2) * Math.log(beta / (2 * Math.PI)) - (beta / 2) * (rss + trace);
  const prior = (m / 2) * Math.log(alpha / (2 * Math.PI)) - (alpha / 2) * (mm + ss);
  return lik + prior + ent;
}

/** Log marginal likelihood ln p(y | α, β) of Bayesian linear regression (Bishop 3.86). */
export function logEvidence(Phi: Mat, y: Vec, alpha: number, beta: number, mean: Vec, A: Mat): number {
  const n = y.length, m = mean.length;
  let rss = 0;
  for (let k = 0; k < n; k++) { const r = (y[k] ?? 0) - dot(Phi[k] ?? [], mean); rss += r * r; }
  const E = (beta / 2) * rss + (alpha / 2) * dot(mean, mean);
  return (m / 2) * Math.log(alpha) + (n / 2) * Math.log(beta) - E - 0.5 * logDet(A) - (n / 2) * Math.log(2 * Math.PI);
}

/**
 * Mean-field VI: the best fully-factorised Gaussian q(w) = Π N(μ_i, s_i²) — the
 * family Bayes-by-Backprop trains by SGD — computed at its optimum in closed form.
 * For a Gaussian posterior with precision A = αI + βΦᵀΦ the ELBO is maximised by
 * s_i² = 1/A_ii and μ solving Aμ = βΦᵀy, i.e. μ = m, the exact posterior mean
 * (the fixed point of coordinate ascent; Bishop §10.1.2). Every s_i² ≤ Σ_ii, and
 * KL(q‖p) = ½(Σ log A_ii − log|A|) is the ELBO's gap to the log evidence.
 */
export function meanFieldVI(Phi: Mat, y: Vec, alpha: number, beta: number, A: Mat, exactMean: Vec, exactCov: Mat): MeanField {
  const mu = exactMean.slice();
  const s2 = diagOf(A).map((a) => 1 / a);
  const e = elbo(Phi, y, alpha, beta, mu, s2);
  const lz = logEvidence(Phi, y, alpha, beta, exactMean, A);
  const ratios = s2.map((v, i) => v / (exactCov[i]?.[i] ?? 1)).sort((a, b) => a - b);
  const varRatio = {
    min: ratios[0] ?? 1,
    median: ratios[Math.floor(ratios.length / 2)] ?? 1,
    max: ratios[ratios.length - 1] ?? 1,
  };
  return { mu, s2, elbo: e, logEvidence: lz, kl: lz - e, varRatio };
}

/**
 * Output weights fitted under dropout. With inverted-dropout masks m_i ~
 * Bernoulli(1−p) on the features, E_mask‖y − Φ(m⊙w)/(1−p)‖² = ‖y − Φw‖² +
 * (p/(1−p))·wᵀdiag(ΦᵀΦ)w, so minimising β/2·(that) + α/2·‖w‖² is a ridge
 * regression with a per-feature penalty: w = (βΦᵀΦ + β·p/(1−p)·diag(ΦᵀΦ) + αI)⁻¹βΦᵀy.
 */
export function dropoutFit(Phi: Mat, y: Vec, alpha: number, beta: number, p: number): Vec {
  const Pt = transpose(Phi);
  const PtP = matMul(Pt, Phi);
  const k = p / (1 - p);
  const A = PtP.map((row, i) => row.map((v, j) => beta * v + (i === j ? beta * k * v + alpha : 0)));
  const rhs = matVec(Pt, y).map((v) => beta * v);
  return matVec(invert(A), rhs);
}

export interface BnnModel {
  alpha: number;
  beta: number;
  phiGrid: Mat;          // NXS × M design on the plotting grid
  exactMean: Vec;        // posterior mean m
  cholCov: Mat;          // Cholesky factor of Σ
  meanCurve: number[];   // Φ_grid m
  exactSd: number[];     // exact epistemic σ_f(x) = √(φᵀΣφ) on the grid
  mf: MeanField;
  mfSd: number[];        // mean-field σ_f(x) = √(Σ φ_i² s_i²)
  wDrop: Vec;            // dropout-fitted weights
  dropMean: number[];    // Φ_grid w_drop
  dropSd: number[];      // analytic mask std √(p/(1−p)·Σ φ_i² w_i²)
  ensemble: number[][];  // K member curves on the grid
}

export function buildBnn(alpha: number, beta: number): BnnModel {
  const feat = makeFeatures(M, FEATURE_SEED);
  const Phi = DATA_SET.xs.map((x) => features(x, feat.w1, feat.b1));
  const phiGrid = XS.map((x) => features(x, feat.w1, feat.b1));
  const post = bayesLinear(Phi, DATA_SET.ys, alpha, beta);
  const Pt = transpose(Phi);
  const A = matMul(Pt, Phi).map((row, i) => row.map((v, j) => beta * v + (i === j ? alpha : 0)));
  const cholCov = cholesky(post.cov, 1e-9);
  const meanCurve = phiGrid.map((phi) => dot(phi, post.mean));
  const exactSd = phiGrid.map((phi) => Math.sqrt(Math.max(0, dot(phi, matVec(post.cov, phi)))));
  const mf = meanFieldVI(Phi, DATA_SET.ys, alpha, beta, A, post.mean, post.cov);
  const mfSd = phiGrid.map((phi) => Math.sqrt(phi.reduce((s, f, i) => s + f * f * (mf.s2[i] ?? 0), 0)));
  const wDrop = dropoutFit(Phi, DATA_SET.ys, alpha, beta, DROPOUT_P);
  const kd = DROPOUT_P / (1 - DROPOUT_P);
  const dropMean = phiGrid.map((phi) => dot(phi, wDrop));
  const dropSd = phiGrid.map((phi) => Math.sqrt(kd * phi.reduce((s, f, i) => s + f * f * (wDrop[i] ?? 0) ** 2, 0)));
  const ensemble: number[][] = [];
  for (let k = 0; k < ENS_K; k++) {
    const fk = makeFeatures(M, ENSEMBLE_SEED0 + k);
    const Pk = DATA_SET.xs.map((x) => features(x, fk.w1, fk.b1));
    const pk = bayesLinear(Pk, DATA_SET.ys, alpha, beta);
    ensemble.push(XS.map((x) => dot(features(x, fk.w1, fk.b1), pk.mean)));
  }
  return { alpha, beta, phiGrid, exactMean: post.mean, cholCov, meanCurve, exactSd, mf, mfSd, wDrop, dropMean, dropSd, ensemble };
}

/** How many curves each mode draws. */
export const targetCount = (mode: BnnMode) => (mode === 'point' ? 1 : mode === 'ensemble' ? ENS_K : TARGET_SAMPLES);

/** Draw the i-th curve of a mode on the grid (`rand` is a U(0,1) source). */
export function drawCurve(model: BnnModel, mode: BnnMode, i: number, rand: () => number): number[] {
  if (mode === 'point') return model.meanCurve.slice();
  if (mode === 'ensemble') return (model.ensemble[i % ENS_K] ?? model.meanCurve).slice();
  let w: Vec;
  if (mode === 'exact') {
    const z = model.exactMean.map(() => gaussFrom(rand));
    w = model.exactMean.map((mv, r) => mv + dot(model.cholCov[r] ?? [], z));
  } else if (mode === 'variational') {
    w = model.mf.mu.map((mv, r) => mv + Math.sqrt(model.mf.s2[r] ?? 0) * gaussFrom(rand));
  } else {
    // MC-dropout: an inverted-dropout mask on the features of the dropout-fitted net
    w = model.wDrop.map((wv) => (rand() > DROPOUT_P ? wv / (1 - DROPOUT_P) : 0));
  }
  return model.phiGrid.map((phi) => dot(phi, w));
}

/** Pointwise mean and (population) standard deviation across drawn curves. */
export function curveStats(curves: number[][]): { mean: number[]; sd: number[] } {
  const n = curves.length;
  const mean = new Array(NXS).fill(0), sd = new Array(NXS).fill(0);
  if (n === 0) return { mean, sd };
  for (let g = 0; g < NXS; g++) {
    let m = 0; for (const c of curves) m += c[g] ?? 0; m /= n;
    let v = 0; for (const c of curves) { const d = (c[g] ?? 0) - m; v += d * d; }
    mean[g] = m; sd[g] = Math.sqrt(v / n);
  }
  return { mean, sd };
}
