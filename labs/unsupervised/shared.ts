// Math helpers shared by the Unsupervised Learning labs.
export interface UPt { x: number; y: number; }

/** Eigen-decomposition of a symmetric 2×2 covariance [[a,b],[b,c]]. */
export function eig2(a: number, b: number, c: number) {
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-9, tr / 2 - disc);
  let vx: number, vy: number;
  if (Math.abs(b) > 1e-12) { vx = b; vy = l1 - a; } else { vx = a >= c ? 1 : 0; vy = a >= c ? 0 : 1; }
  const n = Math.hypot(vx, vy) || 1;
  return { l1, l2, angle: Math.atan2(vy / n, vx / n) };
}

/**
 * log 𝒩((px,py) | (mx,my), [[a,b],[b,c]]). Works in log space so a point far
 * from every component never underflows to 0; a covariance that is not
 * positive definite (det ≤ 0) has no density → −∞.
 */
export function logGauss2(px: number, py: number, mx: number, my: number, a: number, b: number, c: number) {
  const det = a * c - b * b;
  if (!(det > 0) || !(a > 0)) return -Infinity;
  const dx = px - mx, dy = py - my;
  const m = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det; // squared Mahalanobis distance
  return -0.5 * m - Math.log(2 * Math.PI) - 0.5 * Math.log(det);
}

export const dist2 = (a: UPt, b: UPt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
