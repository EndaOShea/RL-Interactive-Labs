// Shared, deterministic small-matrix maths for the Stochastic & Bayesian Models
// labs (Bayesian NN, Gaussian Process, HMM). Everything is analytic and
// client-side — tiny hand-rolled linear algebra (matrices ≤ ~60×60) so the labs
// stay correct and fast with no SGD, no TF.js, no servers.

export type Vec = number[];
export type Mat = number[][];

/* ---------- seeded PRNG (mulberry32) + Gaussian ---------- */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Standard-normal sample from a uniform generator (Box–Muller). */
export function gaussFrom(r: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- tiny vector / matrix ops ---------- */
export const zeros = (n: number): Vec => new Array(n).fill(0);
export const zerosM = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array(c).fill(0));
export const idn = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
export const addV = (a: Vec, b: Vec): Vec => a.map((x, i) => x + b[i]);
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * b[i], 0);
export const matVec = (M: Mat, v: Vec): Vec => M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
export const transpose = (A: Mat): Mat => A[0].map((_, j) => A.map((row) => row[j]));
export function matMul(A: Mat, B: Mat): Mat {
  const n = A.length, m = B[0].length, k = B.length;
  const C = zerosM(n, m);
  for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; for (let j = 0; j < m; j++) C[i][j] += a * B[p][j]; }
  return C;
}
/** A + λI (returns a copy). */
export const addDiag = (A: Mat, lambda: number): Mat => A.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));

/** Gauss–Jordan inverse of an n×n matrix with partial pivoting. */
export function invert(A: Mat): Mat {
  const n = A.length;
  const I = idn(n);
  const M = A.map((row, i) => [...row, ...I[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) M[piv][col] += 1e-10;   // guard singularity
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}
export const solve = (A: Mat, b: Vec): Vec => matVec(invert(A), b);

/** Lower-triangular Cholesky factor L with L·Lᵀ = A (+ jitter for stability). */
export function cholesky(A: Mat, jitter = 1e-9): Mat {
  const n = A.length;
  const L = zerosM(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(A[i][i] + jitter - s, 1e-12));
      else L[i][j] = (A[i][j] - s) / (L[j][j] || 1e-12);
    }
  }
  return L;
}
/** Sample from N(mean, cov) given the Cholesky factor L of cov: mean + L·z. */
export function mvnSample(mean: Vec, L: Mat, r: () => number): Vec {
  const z = mean.map(() => gaussFrom(r));
  return addV(mean, matVec(L, z));
}

/* ---------- GP kernels (1-D) ---------- */
export type KernelId = 'rbf' | 'matern32' | 'periodic' | 'linear';
export const KERNELS: { id: KernelId; label: string }[] = [
  { id: 'rbf', label: 'RBF (smooth)' },
  { id: 'matern32', label: 'Matérn-3/2 (rough)' },
  { id: 'periodic', label: 'periodic' },
  { id: 'linear', label: 'linear' },
];

export function kernel(id: KernelId, x1: number, x2: number, ell: number, sf: number, period = 0.3): number {
  const r = Math.abs(x1 - x2);
  const v = sf * sf;
  switch (id) {
    case 'rbf': return v * Math.exp(-(r * r) / (2 * ell * ell));
    case 'matern32': { const a = (Math.sqrt(3) * r) / ell; return v * (1 + a) * Math.exp(-a); }
    case 'periodic': { const s = Math.sin((Math.PI * r) / period); return v * Math.exp((-2 * s * s) / (ell * ell)); }
    case 'linear': return v * ((x1 - 0.5) * (x2 - 0.5)) + 0.02 * v;
  }
}
/** Gram matrix k(X, X) for a 1-D input set. */
export function gram(id: KernelId, X: Vec, ell: number, sf: number, period = 0.3): Mat {
  const n = X.length;
  const K = zerosM(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i][j] = kernel(id, X[i], X[j], ell, sf, period);
  return K;
}
/** Cross-covariance k(Xa, Xb). */
export function cross(id: KernelId, Xa: Vec, Xb: Vec, ell: number, sf: number, period = 0.3): Mat {
  return Xa.map((xa) => Xb.map((xb) => kernel(id, xa, xb, ell, sf, period)));
}

/* ---------- Bayesian-NN random features (fixed hidden layer) ---------- */
/**
 * A BNN with a fixed random first layer + a Bayesian linear output layer is
 * exactly Bayesian linear regression in feature space — the principled, training-
 * free core all four inference modes approximate. φ(x) = tanh(w1·x + b1).
 */
export function makeFeatures(M: number, seed: number): { w1: Vec; b1: Vec } {
  const r = rng(seed);
  const w1 = Array.from({ length: M }, () => gaussFrom(r) * 4.5);     // input frequencies
  const b1 = Array.from({ length: M }, () => (r() * 2 - 1) * 3);      // phases
  return { w1, b1 };
}
export const features = (x: number, w1: Vec, b1: Vec): Vec => w1.map((w, m) => Math.tanh(w * x + b1[m]));

/**
 * Bayesian linear regression on a design matrix Φ (n×M) with Gaussian prior
 * precision α and noise precision β. Returns the posterior mean m and covariance
 * Σ over the output weights: Σ = (αI + βΦᵀΦ)⁻¹, m = βΣΦᵀy.
 */
export function bayesLinear(Phi: Mat, y: Vec, alpha: number, beta: number): { mean: Vec; cov: Mat } {
  const M = Phi[0].length;
  const Pt = transpose(Phi);              // M×n
  const PtP = matMul(Pt, Phi);            // M×M
  const A = PtP.map((row, i) => row.map((v, j) => beta * v + (i === j ? alpha : 0)));
  const cov = invert(A);                  // Σ
  const Pty = matVec(Pt, y);              // M
  const mean = matVec(cov, Pty).map((v) => beta * v);
  return { mean, cov };
}
