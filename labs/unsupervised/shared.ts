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

/** 2-D Gaussian density at (px,py) with mean (mx,my) and covariance [[a,b],[b,c]]. */
export function gauss2(px: number, py: number, mx: number, my: number, a: number, b: number, c: number) {
  const det = a * c - b * b || 1e-9;
  const inv = 1 / det;
  const dx = px - mx, dy = py - my;
  const m = inv * (c * dx * dx - 2 * b * dx * dy + a * dy * dy);
  return Math.exp(-0.5 * m) / (2 * Math.PI * Math.sqrt(det));
}

export const dist2 = (a: UPt, b: UPt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
