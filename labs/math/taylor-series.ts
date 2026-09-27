// Exact maths for the Taylor Series lab: closed-form / recurrence Taylor
// coefficients, the radius of convergence about any centre a, and the [m/m]
// Padé approximant built from the same coefficients. No numerical
// differentiation anywhere — every coefficient is exact to floating point.

export type TaylorFn = 'sin' | 'cos' | 'exp' | 'geom' | 'log' | 'tanh' | 'runge';

export interface TaylorFnDef {
  label: string;
  f: (x: number) => number;
  /** Taylor coefficients c_0..c_deg about the centre a (c_k = f⁽ᵏ⁾(a)/k!). */
  coeffs: (a: number, deg: number) => number[];
  /** Radius of convergence about a = distance to the nearest (complex) singularity. */
  radius: (a: number) => number;
  /** How that radius is found, for the given centre (display text). */
  radiusText: (a: number) => string;
  /** The singularity that sets the radius (display text). */
  singularity: string;
  domain: [number, number];
  note: string;
}

const fact = (n: number): number => { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; };

/** Coefficients of a function whose derivatives cycle with period 4 (sin, cos). */
const cyclic = (d: [number, number, number, number], deg: number): number[] => {
  const c: number[] = [];
  for (let k = 0; k <= deg; k++) c.push((d[k % 4] ?? 0) / fact(k));
  return c;
};

/** tanh about a: T′ = 1 − T², so (k+1)·c_{k+1} = [k = 0] − Σ_{i+j=k} c_i c_j (Cauchy product). */
export function tanhCoeffs(a: number, deg: number): number[] {
  const c: number[] = [Math.tanh(a)];
  for (let k = 0; k < deg; k++) {
    let conv = 0;
    for (let i = 0; i <= k; i++) conv += (c[i] ?? 0) * (c[k - i] ?? 0);
    c.push(((k === 0 ? 1 : 0) - conv) / (k + 1));
  }
  return c;
}

/**
 * Runge 1/(1+25x²) about a: with x = a + t, (1 + 25a² + 50a·t + 25t²)·Σ c_k t^k = 1, so
 * c_0 = 1/(1+25a²), c_1 = −50a·c_0/(1+25a²), c_k = −(50a·c_{k−1} + 25·c_{k−2})/(1+25a²).
 */
export function rungeCoeffs(a: number, deg: number): number[] {
  const D = 1 + 25 * a * a;
  const c: number[] = [1 / D];
  for (let k = 1; k <= deg; k++) {
    const prev = c[k - 1] ?? 0;
    const prev2 = k >= 2 ? (c[k - 2] ?? 0) : 0;
    c.push(-(50 * a * prev + 25 * prev2) / D);
  }
  return c;
}

const fmtR = (r: number) => r.toFixed(3);

export const TAYLOR_FNS: Record<TaylorFn, TaylorFnDef> = {
  sin: {
    label: 'sin x',
    f: Math.sin,
    coeffs: (a, deg) => cyclic([Math.sin(a), Math.cos(a), -Math.sin(a), -Math.cos(a)], deg),
    radius: () => Infinity,
    radiusText: () => 'R = ∞ (sin is entire — no singularity anywhere)',
    singularity: 'no singularity (entire)',
    domain: [-7, 7],
    note: 'sin is entire — its Taylor series converges everywhere, though far from a you need many terms.',
  },
  cos: {
    label: 'cos x',
    f: Math.cos,
    coeffs: (a, deg) => cyclic([Math.cos(a), -Math.sin(a), -Math.cos(a), Math.sin(a)], deg),
    radius: () => Infinity,
    radiusText: () => 'R = ∞ (cos is entire — no singularity anywhere)',
    singularity: 'no singularity (entire)',
    domain: [-7, 7],
    note: 'cos is entire; its even-powered series is the backbone of countless approximations.',
  },
  exp: {
    label: 'eˣ',
    f: Math.exp,
    coeffs: (a, deg) => {
      const ea = Math.exp(a);
      const c: number[] = [];
      for (let k = 0; k <= deg; k++) c.push(ea / fact(k));
      return c;
    },
    radius: () => Infinity,
    radiusText: () => 'R = ∞ (eˣ is entire — no singularity anywhere)',
    singularity: 'no singularity (entire)',
    domain: [-3, 3],
    note: 'eˣ equals its own derivative, so every coefficient is eᵃ/n! — the series converges everywhere.',
  },
  geom: {
    label: '1/(1−x)',
    f: (x) => 1 / (1 - x),
    coeffs: (a, deg) => {
      const c: number[] = [];
      for (let k = 0; k <= deg; k++) c.push(1 / (1 - a) ** (k + 1));
      return c;
    },
    radius: (a) => Math.abs(1 - a),
    radiusText: (a) => `R = |1 − a| = ${fmtR(Math.abs(1 - a))} (pole at x = 1)`,
    singularity: 'the pole at x = 1',
    domain: [-2, 0.95],
    note: 'The geometric series. Its radius of convergence is the distance to the pole at x = 1, so about a = 0 it only converges for |x| < 1.',
  },
  log: {
    label: 'ln(1+x)',
    f: (x) => Math.log(1 + x),
    coeffs: (a, deg) => {
      const c: number[] = [Math.log(1 + a)];
      for (let k = 1; k <= deg; k++) c.push(((-1) ** (k - 1)) / (k * (1 + a) ** k));
      return c;
    },
    radius: (a) => Math.abs(1 + a),
    radiusText: (a) => `R = 1 + a = ${fmtR(Math.abs(1 + a))} (branch point at x = −1)`,
    singularity: 'the branch point at x = −1',
    domain: [-0.9, 3],
    note: 'ln(1+x) has a singularity at x = −1, so about a = 0 its series only converges on (−1, 1].',
  },
  tanh: {
    label: 'tanh x',
    f: Math.tanh,
    coeffs: tanhCoeffs,
    radius: (a) => Math.hypot(a, Math.PI / 2),
    radiusText: (a) => `R = √(a² + (π/2)²) = ${fmtR(Math.hypot(a, Math.PI / 2))} (complex poles at ±iπ/2)`,
    singularity: 'the complex poles at ±iπ/2',
    domain: [-3.4, 3.4],
    note: 'tanh saturates to ±1. Its series has a finite radius because of the complex poles at ±iπ/2 — about 0 it only converges for |x| < π/2, even though tanh is bounded and smooth on the real line.',
  },
  runge: {
    label: '1/(1+25x²)',
    f: (x) => 1 / (1 + 25 * x * x),
    coeffs: rungeCoeffs,
    radius: (a) => Math.hypot(a, 0.2),
    radiusText: (a) => `R = √(a² + 1/25) = ${fmtR(Math.hypot(a, 0.2))} (complex poles at ±i/5)`,
    singularity: 'the complex poles at ±i/5',
    domain: [-1, 1],
    note: "Runge's function: a smooth bell with complex poles at ±i/5, so its series about 0 only converges on |x| < 0.2. High-degree polynomials oscillate wildly near the edges — the classic Runge phenomenon.",
  },
};

/** Polynomial Σ coef[k]·t^k by Horner's rule. */
export function polyEval(coef: number[], t: number): number {
  let s = 0;
  for (let k = coef.length - 1; k >= 0; k--) s = s * t + (coef[k] ?? 0);
  return s;
}

/** Tₙ(x) = Σ_{k≤n} c_k (x − a)^k. */
export const taylorEval = (c: number[], a: number, n: number, x: number): number =>
  polyEval(c.slice(0, n + 1), x - a);

/** Relative pivot threshold below which a Padé Toeplitz system is treated as singular. */
export const PADE_SINGULAR_TOL = 1e-10;

/**
 * Solve A·z = rhs by Gaussian elimination with partial pivoting. Returns null when a
 * pivot falls below tol·max|A| (the system is singular to working precision).
 */
export function solveLinear(A: number[][], rhs: number[], tol = PADE_SINGULAR_TOL): number[] | null {
  const n = rhs.length;
  const M = A.map((row, i) => [...row, rhs[i] ?? 0]);
  let scale = 0;
  for (const row of A) for (const v of row) scale = Math.max(scale, Math.abs(v));
  if (scale === 0) return null;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    if (Math.abs(M[piv]![col]!) <= tol * scale) return null;
    [M[col], M[piv]] = [M[piv]!, M[col]!];
    const pr = M[col]!;
    for (let r = col + 1; r < n; r++) {
      const row = M[r]!;
      const f = row[col]! / pr[col]!;
      for (let k = col; k <= n; k++) row[k] = row[k]! - f * pr[k]!;
    }
  }
  const z = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    const row = M[i]!;
    let s = row[n]!;
    for (let k = i + 1; k < n; k++) s -= row[k]! * z[k]!;
    z[i] = s / row[i]!;
  }
  return z;
}

export interface PadeFit {
  /** Order asked for: [m/m] with m = ⌊n/2⌋. */
  requested: number;
  /** Order actually built — lowered while the [m/m] Toeplitz system is singular. */
  m: number;
  /** Numerator P coefficients in t = x − a. */
  p: number[];
  /** Denominator Q coefficients in t = x − a (q[0] = 1). */
  q: number[];
}

/**
 * The [m/m] Padé approximant P/Q from Taylor coefficients c[0..2m]: Q (q₀ = 1) solves
 * Σ_{j=0}^{m} q_j c_{m+i−j} = 0 for i = 1..m, then p_i = Σ_{k≤min(i,m)} c_{i−k} q_k.
 * When that m×m Toeplitz system is singular (e.g. 1/(1−x) is already rational, so
 * every [m/m] with m ≥ 1 collapses to [1/1]) the order is lowered until it is solvable
 * — never silently replaced by the Taylor polynomial.
 */
export function padeDiag(c: number[], requested: number): PadeFit {
  for (let m = Math.min(requested, Math.floor((c.length - 1) / 2)); m >= 1; m--) {
    const A: number[][] = [];
    const rhs: number[] = [];
    for (let i = 0; i < m; i++) {
      const row: number[] = [];
      for (let j = 0; j < m; j++) row.push(c[m + i - j] ?? 0);
      A.push(row);
      rhs.push(-(c[m + i + 1] ?? 0));
    }
    const b = solveLinear(A, rhs);
    if (!b) continue;
    const q = [1, ...b];
    const p: number[] = [];
    for (let i = 0; i <= m; i++) {
      let s = 0;
      for (let k = 0; k <= Math.min(i, m); k++) s += (c[i - k] ?? 0) * (q[k] ?? 0);
      p.push(s);
    }
    return { requested, m, p, q };
  }
  return { requested, m: 0, p: [c[0] ?? 0], q: [1] };
}

/** R(x) = P(x − a)/Q(x − a); NaN where Q vanishes (a pole of the approximant). */
export function padeEval(fit: PadeFit, a: number, x: number): number {
  const t = x - a;
  const den = polyEval(fit.q, t);
  let mag = 0;
  for (let k = 0; k < fit.q.length; k++) mag += Math.abs(fit.q[k] ?? 0) * Math.abs(t) ** k;
  if (Math.abs(den) <= 1e-12 * mag) return NaN;
  return polyEval(fit.p, t) / den;
}

/** Padé curve samples, with a NaN break wherever Q changes sign between samples (a pole). */
export function padeCurve(fit: PadeFit, a: number, xs: number[]): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  let prevQ: number | null = null;
  let prevX = 0;
  for (const x of xs) {
    const qv = polyEval(fit.q, x - a);
    if (prevQ !== null && Math.sign(qv) !== Math.sign(prevQ) && qv !== 0 && prevQ !== 0) {
      out.push({ x: (x + prevX) / 2, y: NaN });
    }
    out.push({ x, y: padeEval(fit, a, x) });
    prevQ = qv;
    prevX = x;
  }
  return out;
}

export type RadiusCase = 'inside' | 'boundary' | 'outside';

/** Where the eval point sits relative to the radius of convergence about a. */
export function radiusCase(R: number, dist: number): RadiusCase {
  if (!Number.isFinite(R)) return 'inside';
  if (Math.abs(dist - R) <= 1e-9) return 'boundary';
  return dist < R ? 'inside' : 'outside';
}

/** The lab's order for the approximant at degree n: Taylor Tₙ, or Padé [⌊n/2⌋/⌊n/2⌋]. */
export const padeOrder = (n: number): number => Math.floor(n / 2);

export const TAYLOR_MAX_CAP = 10;
