// PCA maths for the PCA lab. Pure module.
// Covariance uses the population normaliser 1/n (as the export does via
// np.cov(..., bias=True)); the 2×2 eigen-decomposition is closed form.
import type { Rng } from './rng';
import { gauss, q4 } from './rng';

export interface XYp { x: number; y: number; }

export interface Pca {
  mx: number; my: number;
  cxx: number; cyy: number; cxy: number;
  l1: number; l2: number;       // eigenvalues, l1 ≥ l2
  v1x: number; v1y: number;     // unit PC1 (PC2 = (−v1y, v1x))
  e1: number; e2: number;       // explained-variance shares
}

export function computePCA(pts: XYp[], ref?: XYp): Pca {
  const n = pts.length || 1;
  const mx = pts.reduce((s, p) => s + p.x, 0) / n;
  const my = pts.reduce((s, p) => s + p.y, 0) / n;
  let cxx = 0, cyy = 0, cxy = 0;
  pts.forEach((p) => { const dx = p.x - mx, dy = p.y - my; cxx += dx * dx; cyy += dy * dy; cxy += dx * dy; });
  cxx /= n; cyy /= n; cxy /= n;
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  let v1x: number, v1y: number;
  if (Math.abs(cxy) > 1e-12) { v1x = cxy; v1y = l1 - cxx; } else { v1x = cxx >= cyy ? 1 : 0; v1y = cxx >= cyy ? 0 : 1; }
  const norm = Math.hypot(v1x, v1y) || 1; v1x /= norm; v1y /= norm;
  // An eigenvector's sign is arbitrary; point PC1 along `ref` so it does not flip as the cloud rotates.
  if (ref && v1x * ref.x + v1y * ref.y < 0) { v1x = -v1x; v1y = -v1y; }
  const total = l1 + l2 || 1;
  return { mx, my, cxx, cyy, cxy, l1, l2, v1x, v1y, e1: l1 / total, e2: l2 / total };
}

/** Scores t = Vᵀ(x − μ) (coordinates along PC1, PC2). */
export const scores = (pts: XYp[], p: Pca) => pts.map((q) => {
  const dx = q.x - p.mx, dy = q.y - p.my;
  return { t1: dx * p.v1x + dy * p.v1y, t2: -dx * p.v1y + dy * p.v1x };
});

/** Whitened scores z = Λ^{-1/2}·Vᵀ(x − μ): identity covariance by construction. */
export const whitened = (pts: XYp[], p: Pca) => scores(pts, p).map(({ t1, t2 }) => ({
  z1: t1 / Math.sqrt(Math.max(p.l1, 1e-300)), z2: t2 / Math.sqrt(Math.max(p.l2, 1e-300)),
}));

/** Rank-1 reconstruction x̂ = μ + t₁·v₁ and its mean squared error (= λ₂ exactly). */
export function reconstruct(pts: XYp[], p: Pca) {
  const hat = scores(pts, p).map(({ t1 }) => ({ x: p.mx + t1 * p.v1x, y: p.my + t1 * p.v1y }));
  const mse = pts.reduce((s, q, i) => s + (q.x - hat[i]!.x) ** 2 + (q.y - hat[i]!.y) ** 2, 0) / (pts.length || 1);
  return { hat, mse };
}

/** Base cloud: n pairs of independent standard normals (rounded to 4 decimals). */
export const makeCloud = (r: Rng, n: number): [number, number][] =>
  Array.from({ length: n }, () => [q4(gauss(r)), q4(gauss(r))] as [number, number]);

/** The lab's data: stretch the base cloud by (0.98, elongation), rotate by `angle`, scale 0.13 about (0.5, 0.5). */
export function shapeCloud(base: [number, number][], angle: number, elong: number): XYp[] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return base.map(([u, v]) => {
    const a = u * 0.98, b = v * elong;
    return { x: 0.5 + (a * c - b * s) * 0.13, y: 0.5 + (a * s + b * c) * 0.13 };
  });
}
