// Pure maths for the Derivatives lab: four functions with closed-form f′, f″, f‴,
// the forward and central difference quotients, and the error-vs-dx curve
// computed by actually evaluating each quotient in double precision (so both the
// truncation slope and the round-off floor are real, not modelled).

export type DerivFnId = 'square' | 'cubic' | 'sin' | 'exp';
export type DiffMethod = 'forward' | 'central';

export interface DerivFn {
  id: DerivFnId;
  label: string;
  expr: string;        // human-readable f(x)
  dexpr: string;       // human-readable f′(x)
  f: (x: number) => number;
  df: (x: number) => number;
  d2f: (x: number) => number;
  d3f: (x: number) => number;
  domain: [number, number];
  range: [number, number];
  defaultX0: number;
  /** f and f′ as Python expressions (for the export). */
  py: { f: string; df: string; name: string };
}

export const DERIV_FNS: DerivFn[] = [
  {
    id: 'square', label: 'x²', expr: 'x²', dexpr: '2x',
    f: (x) => x * x, df: (x) => 2 * x, d2f: () => 2, d3f: () => 0,
    domain: [-2.5, 2.5], range: [-1, 6], defaultX0: 1,
    py: { f: 'x**2', df: '2*x', name: 'x^2' },
  },
  {
    id: 'cubic', label: 'x³−x', expr: 'x³ − x', dexpr: '3x² − 1',
    f: (x) => x * x * x - x, df: (x) => 3 * x * x - 1, d2f: (x) => 6 * x, d3f: () => 6,
    domain: [-1.8, 1.8], range: [-2, 2], defaultX0: 0.8,
    py: { f: 'x**3 - x', df: '3*x**2 - 1', name: 'x^3 - x' },
  },
  {
    id: 'sin', label: 'sin x', expr: 'sin(x)', dexpr: 'cos(x)',
    f: (x) => Math.sin(x), df: (x) => Math.cos(x), d2f: (x) => -Math.sin(x), d3f: (x) => -Math.cos(x),
    domain: [-Math.PI, Math.PI], range: [-1.4, 1.4], defaultX0: 0.6,
    py: { f: 'np.sin(x)', df: 'np.cos(x)', name: 'sin(x)' },
  },
  {
    id: 'exp', label: 'eˣ', expr: 'eˣ', dexpr: 'eˣ',
    f: (x) => Math.exp(x), df: (x) => Math.exp(x), d2f: (x) => Math.exp(x), d3f: (x) => Math.exp(x),
    domain: [-2, 2], range: [-0.5, 7.4], defaultX0: 0.5,
    py: { f: 'np.exp(x)', df: 'np.exp(x)', name: 'e^x' },
  },
];

/** dx slider / sweep limits, in log₁₀ dx. */
export const DX_LOG_MIN = -12;
export const DX_LOG_MAX = Math.log10(2);
/** Points on the computed error curve. */
export const DX_CURVE_N = 250;
/** Unit round-off of IEEE double precision. */
export const MACHINE_EPS = 2 ** -52;
/** Floor for plotting log₁₀|error| when the quotient is exact (error 0). */
export const ERR_FLOOR = 1e-17;

/** dx from its log₁₀, rounded to 6 significant digits (so the export reproduces it exactly). */
export const dxOf = (logDx: number): number => Number((10 ** logDx).toPrecision(6));

/** Forward [f(x+dx) − f(x)]/dx or central [f(x+dx) − f(x−dx)]/(2dx), in double precision. */
export function diffQuotient(fn: DerivFn, x0: number, dx: number, method: DiffMethod): number {
  return method === 'forward'
    ? (fn.f(x0 + dx) - fn.f(x0)) / dx
    : (fn.f(x0 + dx) - fn.f(x0 - dx)) / (2 * dx);
}

/**
 * Leading truncation term of the Taylor expansion: forward ≈ ½|f″(x₀)|·dx,
 * central ≈ |f‴(x₀)|·dx²/6 (0 when that derivative vanishes, e.g. central on x²).
 */
export const truncationModel = (fn: DerivFn, x0: number, dx: number, method: DiffMethod): number =>
  (method === 'forward' ? 0.5 * Math.abs(fn.d2f(x0)) * dx : (Math.abs(fn.d3f(x0)) * dx * dx) / 6);

/** Order-of-magnitude round-off in the quotient: ε·|f(x₀)|/dx (the subtraction loses digits as dx shrinks). */
export const roundoffModel = (fn: DerivFn, x0: number, dx: number): number =>
  (MACHINE_EPS * Math.max(Math.abs(fn.f(x0)), 1e-300)) / dx;

export interface ErrPoint { logDx: number; dx: number; err: number; logErr: number; }

/** The computed |quotient − f′(x₀)| across dx = 10^DX_LOG_MIN … 2 (log-spaced). */
export function errorCurve(fn: DerivFn, x0: number, method: DiffMethod, n = DX_CURVE_N): ErrPoint[] {
  const exact = fn.df(x0);
  const out: ErrPoint[] = [];
  for (let i = 0; i < n; i++) {
    const logDx = DX_LOG_MIN + (i / (n - 1)) * (DX_LOG_MAX - DX_LOG_MIN);
    const dx = dxOf(logDx);
    const err = Math.abs(diffQuotient(fn, x0, dx, method) - exact);
    out.push({ logDx: Math.log10(dx), dx, err, logErr: Math.log10(Math.max(err, ERR_FLOOR)) });
  }
  return out;
}

/** The point of smallest computed error on a curve (the practical optimum dx). */
export function bestOf(curve: ErrPoint[]): ErrPoint | null {
  let best: ErrPoint | null = null;
  for (const p of curve) if (!best || p.err < best.err) best = p;
  return best;
}

/**
 * Observed order p from the computed errors one decade apart (E(10·dx)/E(dx) = 10^p).
 * Only meaningful while truncation dominates; returns NaN when either error is 0.
 */
export function observedOrder(fn: DerivFn, x0: number, dx: number, method: DiffMethod): number {
  const exact = fn.df(x0);
  const e1 = Math.abs(diffQuotient(fn, x0, dx, method) - exact);
  const e10 = Math.abs(diffQuotient(fn, x0, 10 * dx, method) - exact);
  return e1 > 0 && e10 > 0 ? Math.log10(e10 / e1) : NaN;
}
