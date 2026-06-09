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
    const d = Math.abs(def.minima[i] - x);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
