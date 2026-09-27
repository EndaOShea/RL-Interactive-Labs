// Polynomial / ridge regression maths for the Linear Regression lab. Pure module.
//
// x lives on [−1, 1] and the polynomial features are the Legendre polynomials
// Pⱼ(x), j = 1..d (P₁ = x, P₂ = (3x²−1)/2, …). They span exactly the same
// functions as [x, x², …, xᵈ] but are mutually orthogonal under a uniform x and
// bounded by 1, so the loss surface is well-conditioned (Hessian eigenvalues
// ≈ 1/(2j+1), λmax ≈ 1 from the bias) and plain gradient descent converges in
// tens to hundreds of epochs, where raw monomials need tens of thousands.
// A higher degree still costs more weight per unit of wiggle (‖Pⱼ‖² = 1/(2j+1)),
// so the ridge penalty prefers smooth curves.
//   J(w, b)   = ½·mean((ŷ − y)²)                   (reported loss, "½·MSE")
//   objective = J + ½·λ·‖w‖²                        (ridge; the bias b is not penalised)
//   GD step   : w ← w − α(∂J/∂w + λw),  b ← b − α·∂J/∂b
import type { Rng } from './rng';
import { gauss, q4 } from './rng';

/** Legendre coefficients of the true curve f = c0 + c1·P1 + c2·P2 + c3·P3 (a cubic). */
export const TRUTH = { c0: 0.1, c1: 0.5, c2: -0.45, c3: 0.35 };
export const truth = (x: number) =>
  TRUTH.c0 + TRUTH.c1 * x + TRUTH.c2 * (3 * x * x - 1) / 2 + TRUTH.c3 * (5 * x ** 3 - 3 * x) / 2;

export interface XY { x: number; y: number }
export interface Model { w: number[]; b: number }

/** Legendre features P₁(x)..P_d(x). */
export function features(x: number, d: number): number[] {
  const out: number[] = [];
  let p0 = 1, p1 = x;
  for (let j = 1; j <= d; j++) {
    out.push(p1);
    const p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1); // Bonnet recurrence
    p0 = p1; p1 = p2;
  }
  return out;
}

export const predict = (x: number, m: Model) => features(x, m.w.length).reduce((s, f, i) => s + m.w[i]! * f, m.b);

/**
 * Training + held-out test points, y = f(x) + noise·𝒩(0,1), rounded to 4 decimals.
 * Training x are stratified (one uniform draw in each of n equal slices of
 * [−1, 1]) so a small training set still covers the interval; test x ~ U[−1, 1].
 */
export function makeRegData(r: Rng, nTrain: number, nTest: number, noise: number) {
  const at = (x: number): XY => ({ x: q4(x), y: q4(truth(q4(x)) + noise * gauss(r)) });
  const train = Array.from({ length: nTrain }, (_, i) => at(-1 + (2 * (i + r())) / nTrain));
  const test = Array.from({ length: nTest }, () => at(2 * r() - 1));
  return { train, test };
}

/** J = ½·mean((ŷ − y)²). */
export function halfMse(data: XY[], m: Model) {
  if (!data.length) return 0;
  let s = 0;
  for (const p of data) { const e = predict(p.x, m) - p.y; s += e * e; }
  return (0.5 * s) / data.length;
}

export const objective = (data: XY[], m: Model, lam: number) =>
  halfMse(data, m) + 0.5 * lam * m.w.reduce((s, v) => s + v * v, 0);

/** One full-batch gradient-descent epoch. Returns the loss and gradient at the OLD parameters. */
export function gdStep(data: XY[], m: Model, alpha: number, lam: number) {
  const d = m.w.length, n = data.length;
  const dw = new Array<number>(d).fill(0);
  let db = 0, J = 0;
  for (const p of data) {
    const f = features(p.x, d);
    const e = f.reduce((s, v, i) => s + m.w[i]! * v, m.b) - p.y;
    for (let i = 0; i < d; i++) dw[i] = dw[i]! + e * f[i]!;
    db += e; J += e * e;
  }
  for (let i = 0; i < d; i++) dw[i] = dw[i]! / n + lam * m.w[i]!;
  db /= n; J = (0.5 * J) / n;
  const next: Model = { w: m.w.map((wi, i) => wi - alpha * dw[i]!), b: m.b - alpha * db };
  return { next, J, dw, db };
}

/** H = (1/n)·Φ̃ᵀΦ̃ + λ·diag(0,1,…,1) with Φ̃ = [1, P₁..P_d] — the Hessian of the objective in (b, w). */
export function hessian(data: XY[], d: number, lam: number): number[][] {
  const n = Math.max(1, data.length);
  const H = Array.from({ length: d + 1 }, () => new Array<number>(d + 1).fill(0));
  for (const p of data) {
    const row = [1, ...features(p.x, d)];
    for (let i = 0; i <= d; i++) { const Hi = H[i]!; for (let j = 0; j <= d; j++) Hi[j] = Hi[j]! + (row[i]! * row[j]!) / n; }
  }
  for (let i = 1; i <= d; i++) { const Hi = H[i]!; Hi[i] = Hi[i]! + lam; }
  return H;
}

/**
 * Closed-form minimiser of the objective: solve H·θ = (1/n)·Φ̃ᵀy (Gaussian elimination,
 * partial pivoting). With λ = 0 and fewer points than coefficients the minimiser is not
 * unique; a 1e-10 ridge picks the minimum-norm one — the solution GD from zero converges to.
 */
export function closedForm(data: XY[], d: number, lam: number): Model {
  const n = Math.max(1, data.length);
  const A = hessian(data, d, Math.max(lam, 1e-10)).map((r) => [...r]);
  const rhs = new Array<number>(d + 1).fill(0);
  for (const p of data) { const row = [1, ...features(p.x, d)]; for (let i = 0; i <= d; i++) rhs[i] = rhs[i]! + (row[i]! * p.y) / n; }
  const N = d + 1;
  for (let c = 0; c < N; c++) {
    let piv = c;
    for (let r = c + 1; r < N; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[piv]![c]!)) piv = r;
    [A[c], A[piv]] = [A[piv]!, A[c]!]; [rhs[c], rhs[piv]] = [rhs[piv]!, rhs[c]!];
    const a = A[c]![c]!;
    if (Math.abs(a) < 1e-14) continue; // singular direction (fewer points than parameters): leave it at 0
    for (let r = c + 1; r < N; r++) {
      const f = A[r]![c]! / a;
      if (!f) continue;
      const Ar = A[r]!, Ac = A[c]!;
      for (let k = c; k < N; k++) Ar[k] = Ar[k]! - f * Ac[k]!;
      rhs[r] = rhs[r]! - f * rhs[c]!;
    }
  }
  const th = new Array<number>(N).fill(0);
  for (let r = N - 1; r >= 0; r--) {
    const a = A[r]![r]!;
    if (Math.abs(a) < 1e-14) { th[r] = 0; continue; }
    let s = rhs[r]!;
    for (let k = r + 1; k < N; k++) s -= A[r]![k]! * th[k]!;
    th[r] = s / a;
  }
  return { b: th[0]!, w: th.slice(1) };
}

/** Largest and smallest eigenvalues of the (symmetric PSD) Hessian, by cyclic Jacobi rotations. */
export function hessianEigs(data: XY[], d: number, lam: number) {
  const A = hessian(data, d, lam).map((r) => [...r]);
  const N = d + 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < N; p++) for (let q = p + 1; q < N; q++) off += A[p]![q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < N; p++) for (let q = p + 1; q < N; q++) {
      const apq = A[p]![q]!;
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < N; k++) {
        const akp = A[k]![p]!, akq = A[k]![q]!;
        A[k]![p] = c * akp - s * akq; A[k]![q] = s * akp + c * akq;
      }
      for (let k = 0; k < N; k++) {
        const apk = A[p]![k]!, aqk = A[q]![k]!;
        A[p]![k] = c * apk - s * aqk; A[q]![k] = s * apk + c * aqk;
      }
    }
  }
  const ev = A.map((r, i) => r[i]!);
  return { max: Math.max(...ev), min: Math.min(...ev) };
}

/** ½·mean over a fine grid of (ŷ − f)²: how far the fitted curve is from the TRUE curve. */
export function truthGap(m: Model, grid = 201) {
  let s = 0;
  for (let i = 0; i < grid; i++) { const x = -1 + (2 * i) / (grid - 1); const e = predict(x, m) - truth(x); s += e * e; }
  return (0.5 * s) / grid;
}
