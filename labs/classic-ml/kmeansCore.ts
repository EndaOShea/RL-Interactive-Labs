// k-means maths for the k-Means lab (and the k-means++ seeding the GMM lab
// reuses). Pure module, no React. Randomness comes from a seeded Rng.
import type { Rng } from './rng';
import { permutation } from './rng';

export interface P { x: number; y: number; }
export type Init = 'random' | 'kpp' | 'ff';

const d2 = (a: P, b: P) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

/** k-means++ (Arthur & Vassilvitskii): first seed uniform, each next seed sampled ∝ squared distance to the nearest chosen seed. */
export function kppSeeds(pts: P[], k: number, r: Rng): number[] {
  const idx = [Math.floor(r() * pts.length)];
  while (idx.length < k) {
    const dd = pts.map((p) => Math.min(...idx.map((i) => d2(p, pts[i]!))));
    const sum = dd.reduce((a, b) => a + b, 0) || 1;
    let u = r() * sum, pick = 0;
    for (let i = 0; i < pts.length; i++) { u -= dd[i]!; if (u <= 0) { pick = i; break; } }
    idx.push(pick);
  }
  return idx;
}

/**
 * Farthest-first traversal, fully deterministic: the first seed is the point
 * farthest from the data mean, and each next seed is the point farthest from
 * every seed chosen so far (ties → lowest index).
 */
export function farthestFirstSeeds(pts: P[], k: number): number[] {
  const n = pts.length || 1;
  const mean = { x: pts.reduce((s, p) => s + p.x, 0) / n, y: pts.reduce((s, p) => s + p.y, 0) / n };
  let first = 0, fd = -1;
  pts.forEach((p, i) => { const d = d2(p, mean); if (d > fd) { fd = d; first = i; } });
  const idx = [first];
  while (idx.length < k) {
    let bi = 0, bd = -1;
    pts.forEach((p, i) => { const d = Math.min(...idx.map((j) => d2(p, pts[j]!))); if (d > bd) { bd = d; bi = i; } });
    idx.push(bi);
  }
  return idx;
}

/** Seed centroids: random = k distinct points (seeded Fisher–Yates), kpp = k-means++, ff = farthest-first. */
export function initCentroids(pts: P[], k: number, method: Init, r: Rng): P[] {
  const idx = method === 'random' ? permutation(pts.length, r).slice(0, k)
    : method === 'kpp' ? kppSeeds(pts, k, r)
      : farthestFirstSeeds(pts, k);
  return idx.map((i) => ({ x: pts[i]!.x, y: pts[i]!.y }));
}

/** Assignment step: nearest centroid (ties → lowest index). */
export const assign = (pts: P[], cents: P[]) => pts.map((p) => {
  let best = 0, bd = Infinity;
  cents.forEach((c, j) => { const d = d2(p, c); if (d < bd) { bd = d; best = j; } });
  return best;
});

export const inertiaOf = (pts: P[], cents: P[], labels: number[]) =>
  pts.reduce((s, p, i) => s + d2(p, cents[labels[i]!]!), 0);

/** Update step: each centroid moves to its members' mean; an empty cluster keeps its centroid. */
export function updateCentroids(pts: P[], labels: number[], cents: P[]) {
  const sums = cents.map(() => ({ x: 0, y: 0, n: 0 }));
  pts.forEach((p, i) => { const s = sums[labels[i]!]; if (!s) return; s.x += p.x; s.y += p.y; s.n++; });
  const next = cents.map((c, j) => { const s = sums[j]!; return s.n ? { x: s.x / s.n, y: s.y / s.n } : { ...c }; });
  const moved = next.reduce((s, c, j) => s + Math.hypot(c.x - cents[j]!.x, c.y - cents[j]!.y), 0);
  const empty = sums.map((s, j) => (s.n ? -1 : j)).filter((j) => j >= 0);
  return { cents: next, moved, empty };
}

/** Converged when the total centroid movement of an update step falls below this. */
export const KMEANS_TOL = 1e-4;

/** Run Lloyd's algorithm to convergence (assign → update …) exactly as the lab steps it. */
export function runKmeans(pts: P[], init: P[], maxIter = 100) {
  let cents = init;
  let labels = assign(pts, cents);
  let inertia = inertiaOf(pts, cents, labels);
  let iters = 0;
  for (; iters < maxIter; iters++) {
    const u = updateCentroids(pts, labels, cents);
    cents = u.cents;
    labels = assign(pts, cents);
    inertia = inertiaOf(pts, cents, labels);
    if (u.moved < KMEANS_TOL) { iters++; break; }
  }
  return { cents, labels, inertia, iters };
}
