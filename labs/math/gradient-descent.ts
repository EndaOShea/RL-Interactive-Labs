// Pure maths for the Gradient Descent lab: the three 1-D landscapes (with their
// analytic f′ and f″), one optimiser update for each of heavy-ball momentum,
// RMSProp, bias-corrected Adam and Newton, and the stop tests. The lab and its
// Python export both follow exactly these rules.

export type GdFn = 'quadratic' | 'doublewell' | 'wavy';
export type GdOpt = 'momentum' | 'rmsprop' | 'adam' | 'newton';

export interface GdFnDef {
  label: string;
  f: (x: number) => number;
  df: (x: number) => number;
  d2f: (x: number) => number;       // second derivative (curvature / 1-D Hessian)
  domain: [number, number];
  note: string;
  formula: string;
  /** The same three functions as Python expressions (for the export). */
  py: { f: string; df: string; d2f: string };
}

export const GD_FNS: Record<GdFn, GdFnDef> = {
  quadratic: {
    label: 'convex x²',
    f: (x) => x * x,
    df: (x) => 2 * x,
    d2f: () => 2,
    domain: [-3, 3],
    note: 'A convex bowl: one global minimum at x=0. Plain GD converges from anywhere when α < 2/f″ = 1 and diverges when α > 1.',
    formula: 'f(x)=x²',
    py: { f: 'x**2', df: '2*x', d2f: '2.0' },
  },
  doublewell: {
    label: 'double-well x⁴−x²',
    f: (x) => x ** 4 - x * x,
    df: (x) => 4 * x ** 3 - 2 * x,
    d2f: (x) => 12 * x * x - 2,
    domain: [-1.7, 1.7],
    note: 'Two minima at ±√½ and a local max at 0 — a classic non-convex trap. Enough momentum carries the point over the central hump into the other well.',
    formula: 'f(x)=x⁴−x²',
    py: { f: 'x**4 - x**2', df: '4*x**3 - 2*x', d2f: '12*x**2 - 2' },
  },
  wavy: {
    label: 'wavy x²+sin',
    f: (x) => 0.15 * x * x + Math.sin(3 * x),
    df: (x) => 0.3 * x + 3 * Math.cos(3 * x),
    d2f: (x) => 0.3 - 9 * Math.sin(3 * x),
    domain: [-4.2, 4.2],
    note: 'A bowl rippled with sin: many shallow local minima. Where you land depends on the start point and the optimiser.',
    formula: 'f(x)=0.15x²+sin 3x',
    py: { f: '0.15*x**2 + np.sin(3*x)', df: '0.3*x + 3*np.cos(3*x)', d2f: '0.3 - 9*np.sin(3*x)' },
  },
};

export const GD_RHO = 0.9;         // RMSProp decay
export const GD_B1 = 0.9;          // Adam first-moment decay
export const GD_B2 = 0.999;        // Adam second-moment decay
export const GD_EPS = 1e-8;
export const GD_TOL = 1e-3;        // |f′(x′)| (and |Δx| for first-order methods) below this ⇒ stop
export const GD_DIVERGE_MARGIN = 5; // x more than this far outside the domain ⇒ diverged
export const GD_FD_H = 1e-5;       // central-difference step for the gradient check

export interface GdState {
  x: number;
  v: number;   // last signed step Δx (heavy-ball velocity for momentum)
  m: number;   // Adam first moment
  s: number;   // RMSProp / Adam second moment
  t: number;   // steps taken
}

export const gdInit = (x0: number): GdState => ({ x: x0, v: 0, m: 0, s: 0, t: 0 });

export interface GdStepResult {
  next: GdState;
  g: number;              // f′ at the old x
  dx: number;             // signed step taken
  gNext: number;          // f′ at the new x
  converged: boolean;
  diverged: boolean;
  newtonFallback: boolean; // Newton hit |f″| ≤ 1e-8 and took an α-scaled gradient step instead
}

/** One optimiser update from state st. */
export function gdStep(st: GdState, fn: GdFn, opt: GdOpt, alpha: number, beta: number): GdStepResult {
  const def = GD_FNS[fn];
  const g = def.df(st.x);
  let { v, m, s } = st;
  let dx: number;
  let newtonFallback = false;
  if (opt === 'newton') {
    const h = def.d2f(st.x);
    if (Math.abs(h) > 1e-8) dx = -g / h;             // x ← x − f′/f″
    else { dx = -alpha * g; newtonFallback = true; }
    v = dx;
  } else if (opt === 'rmsprop') {
    s = GD_RHO * s + (1 - GD_RHO) * g * g;
    dx = -(alpha * g) / (Math.sqrt(s) + GD_EPS);      // x ← x − α·g/(√s + ε)
    v = dx;
  } else if (opt === 'adam') {
    m = GD_B1 * m + (1 - GD_B1) * g;
    s = GD_B2 * s + (1 - GD_B2) * g * g;
    const t1 = st.t + 1;
    const mhat = m / (1 - GD_B1 ** t1);
    const shat = s / (1 - GD_B2 ** t1);
    dx = -(alpha * mhat) / (Math.sqrt(shat) + GD_EPS); // x ← x − α·m̂/(√ŝ + ε)
    v = dx;
  } else {
    v = beta * st.v - alpha * g;                        // heavy ball (β = 0 ⇒ plain GD)
    dx = v;
  }
  const nx = st.x + dx;
  const [lo, hi] = def.domain;
  const diverged = !Number.isFinite(nx) || nx < lo - GD_DIVERGE_MARGIN || nx > hi + GD_DIVERGE_MARGIN;
  const gNext = def.df(nx);
  // Newton jumps straight to a stationary point, so its test is the gradient alone;
  // first-order methods must also have stopped moving.
  const converged = !diverged && (opt === 'newton'
    ? Math.abs(gNext) < GD_TOL
    : Math.abs(gNext) < GD_TOL && Math.abs(dx) < GD_TOL);
  return { next: { x: nx, v, m, s, t: st.t + 1 }, g, dx, gNext, converged, diverged, newtonFallback };
}

/** Central-difference check of f′(x). */
export const fdGrad = (fn: GdFn, x: number, h = GD_FD_H): number =>
  (GD_FNS[fn].f(x + h) - GD_FNS[fn].f(x - h)) / (2 * h);

/**
 * Local stability limit of heavy-ball GD on the quadratic model at x: iterates stay
 * bounded iff 0 < α·f″(x) < 2(1+β) (for β = 0 this is the classic α < 2/f″).
 * Infinity where f″ ≤ 0 (no local minimum to be stable about).
 */
export const alphaLimit = (fn: GdFn, x: number, beta: number): number => {
  const h = GD_FNS[fn].d2f(x);
  return h > 0 ? (2 * (1 + beta)) / h : Infinity;
};

export interface GdRun { xs: number[]; steps: number; converged: boolean; diverged: boolean; }

/** Run the optimiser until it converges, diverges or hits maxSteps (for checks / the export). */
export function gdRun(x0: number, fn: GdFn, opt: GdOpt, alpha: number, beta: number, maxSteps = 2000): GdRun {
  let st = gdInit(x0);
  const xs = [x0];
  for (let k = 0; k < maxSteps; k++) {
    const r = gdStep(st, fn, opt, alpha, beta);
    st = r.next;
    xs.push(st.x);
    if (r.diverged) return { xs, steps: st.t, converged: false, diverged: true };
    if (r.converged) return { xs, steps: st.t, converged: true, diverged: false };
  }
  return { xs, steps: st.t, converged: false, diverged: false };
}
