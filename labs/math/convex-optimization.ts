// Self-contained maths for the "Convex vs Non-convex" lab. A 1-D landscape on a
// fixed domain with two presets, plus the analytic gradient and the known
// stationary points (used to label which basin a gradient-descent runner lands
// in). Everything here is exact / computed — no mocked values.

export type Surface = 'convex' | 'nonconvex';

export interface SurfaceDef {
  id: Surface;
  label: string;
  formula: string;       // human-readable f(x)
  gradFormula: string;   // human-readable f'(x)
  f: (x: number) => number;
  df: (x: number) => number;   // analytic first derivative
  d2f: (x: number) => number;  // analytic second derivative (curvature)
  domain: [number, number];
  /** Local minima of f on the domain (x positions), precomputed analytically. */
  minima: number[];
  note: string;
}

// CONVEX: f(x) = x²  — single global minimum at x = 0, f'(x) = 2x.
const convex: SurfaceDef = {
  id: 'convex',
  label: 'convex  f(x)=x²',
  formula: 'f(x) = x²',
  gradFormula: "f'(x) = 2x",
  f: (x) => x * x,
  df: (x) => 2 * x,
  d2f: () => 2,
  domain: [-4, 4],
  minima: [0],
  note: 'A convex bowl: a single global minimum at x=0. Every runner, wherever it starts, slides to the same point — initialisation does not matter.',
};

// NON-CONVEX: f(x) = 0.15·x² + 2·sin(3x) on [-4, 4].
// f'(x) = 0.3·x + 6·cos(3x). This has FOUR distinct local minima (found by
// solving f'(x)=0 and checking f''(x)=0.3 − 18·sin(3x) > 0):
//   x ≈ -2.575 (f≈-0.989),  -0.515 (f≈-1.960, GLOBAL),
//        1.545 (f≈-1.636),   3.605 (f≈-0.018),
// with local maxima / flat saddle-like humps between them. Where a runner ends
// up depends entirely on which basin its start falls into.
const nonconvex: SurfaceDef = {
  id: 'nonconvex',
  label: 'non-convex  f(x)=0.15x²+2sin 3x',
  formula: 'f(x) = 0.15·x² + 2·sin 3x',
  gradFormula: "f'(x) = 0.3·x + 6·cos 3x",
  f: (x) => 0.15 * x * x + 2 * Math.sin(3 * x),
  df: (x) => 0.3 * x + 6 * Math.cos(3 * x),
  d2f: (x) => 0.3 - 18 * Math.sin(3 * x),
  domain: [-4, 4],
  minima: [-2.5751, -0.5152, 1.5447, 3.6048],
  note: 'A bowl rippled by a sine: four distinct local minima. The basin a runner ends in is decided by where it started — different initialisations give different answers.',
};

export const SURFACES: Record<Surface, SurfaceDef> = { convex, nonconvex };

/** Index of the nearest known minimum to x (which basin it settled in). */
export function nearestMinIndex(def: SurfaceDef, x: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < def.minima.length; i++) {
    const d = Math.abs((def.minima[i] ?? 0) - x);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

/**
 * Largest α for which plain GD can settle at every minimum: near a minimum the error
 * is multiplied by 1 − α·f″ per step, so it must stay below 2/f″ there.
 */
export const alphaCritical = (def: SurfaceDef): number =>
  Math.min(...def.minima.map((m) => 2 / def.d2f(m)));

/** |f′(x)| below this ⇒ a runner has settled. */
export const CONVERGE_TOL = 1e-3;
/** Step cap for the basin-of-attraction map (a start that has not settled by then is 'unresolved'). */
export const BASIN_MAX_STEPS = 2000;
/** Samples of the curve (and starts of the basin map). */
export const CURVE_N = 241;

/**
 * One runner update, exactly as the lab steps it: a runner whose |f′| is already below
 * the tolerance stays put; otherwise x ← clip(x − α·f′(x)) and it settles once
 * |f′(x′)| < tolerance.
 */
export function runnerStep(def: SurfaceDef, x: number, alpha: number): { x: number; settled: boolean } {
  const g = def.df(x);
  if (Math.abs(g) < CONVERGE_TOL) return { x, settled: true };
  const [lo, hi] = def.domain;
  const nx = Math.max(lo, Math.min(hi, x - alpha * g));
  return { x: nx, settled: Math.abs(def.df(nx)) < CONVERGE_TOL };
}

/** Run one start to convergence (or maxSteps) with the same rule. */
export function descend(def: SurfaceDef, x0: number, alpha: number, maxSteps = BASIN_MAX_STEPS): { x: number; settled: boolean; steps: number } {
  let x = x0;
  for (let k = 0; k < maxSteps; k++) {
    if (Math.abs(def.df(x)) < CONVERGE_TOL) return { x, settled: true, steps: k };
    x = runnerStep(def, x, alpha).x;
  }
  return { x, settled: Math.abs(def.df(x)) < CONVERGE_TOL, steps: maxSteps };
}

/** The curve's x samples across the domain (shared by the plot and the basin map). */
export function curveXs(def: SurfaceDef, n = CURVE_N): number[] {
  const [lo, hi] = def.domain;
  return Array.from({ length: n }, (_, i) => lo + (i / (n - 1)) * (hi - lo));
}

/** A settled point is a minimum only where the curvature is positive (|f′| ≈ 0 also holds at maxima). */
export const isMinimum = (def: SurfaceDef, x: number): boolean => def.d2f(x) > 0;

/**
 * Basin of attraction of every start in xs under gradient descent with this α:
 * the index of the minimum it settles at, or −1 if it has not settled at a minimum
 * within BASIN_MAX_STEPS (α ≥ 2/f″ at the minima makes it bounce; a chance landing
 * next to a maximum is not a minimum either).
 */
export function basinMap(def: SurfaceDef, alpha: number, xs: number[]): number[] {
  return xs.map((x0) => {
    const r = descend(def, x0, alpha);
    return r.settled && isMinimum(def, r.x) ? nearestMinIndex(def, r.x) : -1;
  });
}

/**
 * The lab's reproducible start positions: evenly spread across the padded domain with a
 * small deterministic jitter keyed on the scatter count `seed`.
 */
export function scatterStartsFor(def: SurfaceDef, n: number, seed: number): number[] {
  const [lo, hi] = def.domain;
  const pad = 0.35;
  const a = lo + pad, b = hi - pad;
  return Array.from({ length: n }, (_, i) => {
    const base = a + (i + 0.5) * (b - a) / n;
    const jitter = (Math.sin((i + 1) * 12.9898 + seed * 7.233) * 43758.5453) % 1;
    return Math.max(a, Math.min(b, base + (jitter - 0.5) * 0.3));
  });
}
