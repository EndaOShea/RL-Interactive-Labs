// Self-contained 2-D linear algebra for the Matrix Multiplication lab — dot
// products and matrix·vector, all computed (no mocked numbers).

export type Vec2 = [number, number];
export type Mat2 = [number, number, number, number]; // row-major: a11, a12, a21, a22

export const dot = (a: Vec2, b: Vec2): number => a[0] * b[0] + a[1] * b[1];

export const norm = (a: Vec2): number => Math.hypot(a[0], a[1]);

/** cos θ between a and b (0 when either is the zero vector). */
export const cosTheta = (a: Vec2, b: Vec2): number => {
  const d = norm(a) * norm(b);
  return d < 1e-9 ? 0 : dot(a, b) / d;
};

/** Angle between a and b in radians, clamped for numerical safety. */
export const angleBetween = (a: Vec2, b: Vec2): number =>
  Math.acos(Math.max(-1, Math.min(1, cosTheta(a, b))));

/** Scalar projection of a onto b: comp_b(a) = a·b / |b|. */
export const scalarProj = (a: Vec2, b: Vec2): number => {
  const nb = norm(b);
  return nb < 1e-9 ? 0 : dot(a, b) / nb;
};

/** Vector projection of a onto b: proj_b(a) = (a·b / |b|²) b. */
export const vectorProj = (a: Vec2, b: Vec2): Vec2 => {
  const nb2 = b[0] * b[0] + b[1] * b[1];
  if (nb2 < 1e-9) return [0, 0];
  const k = dot(a, b) / nb2;
  return [k * b[0], k * b[1]];
};

/** Matrix·vector y = A x, with A row-major [a11,a12,a21,a22]. */
export const matVec = (A: Mat2, x: Vec2): Vec2 => [
  A[0] * x[0] + A[1] * x[1],
  A[2] * x[0] + A[3] * x[1],
];

export const det2 = (A: Mat2): number => A[0] * A[3] - A[1] * A[2];

/** Columns of A — where the basis vectors ê₁, ê₂ land. */
export const colsOf = (A: Mat2): { c1: Vec2; c2: Vec2 } => ({
  c1: [A[0], A[2]],
  c2: [A[1], A[3]],
});
