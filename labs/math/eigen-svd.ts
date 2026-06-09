// Exact analytic 2×2 linear algebra for the Eigenvalues & SVD lab.
// All hand-rolled (no numeric solvers): the eigen-decomposition and SVD of a
// general 2×2 matrix have closed forms, so every number shown is exact.

export interface Mat2 { a: number; b: number; c: number; d: number; } // [[a b],[c d]]
export interface Vec2 { x: number; y: number; }

/** A·v for a 2×2 matrix. */
export const apply = (m: Mat2, v: Vec2): Vec2 => ({
  x: m.a * v.x + m.b * v.y,
  y: m.c * v.x + m.d * v.y,
});

export const norm2 = (v: Vec2): number => Math.hypot(v.x, v.y);
export const unit = (v: Vec2): Vec2 => {
  const n = norm2(v);
  return n < 1e-12 ? { x: 0, y: 0 } : { x: v.x / n, y: v.y / n };
};

export interface EigenPair { lambda: number; vec: Vec2; } // vec is unit-length

export interface EigenResult {
  trace: number;
  det: number;
  disc: number;            // t² − 4·det
  complex: boolean;        // true ⇒ rotation, no real invariant axis
  pairs: EigenPair[];      // real eigenpairs (length 2, or 0 when complex)
}

/**
 * Eigenvalues/eigenvectors of a 2×2 matrix via the characteristic equation
 * λ² − t·λ + det = 0, with t = trace, det = ad − bc.
 *   • disc = t² − 4·det ≥ 0  → two real eigenvalues λ = (t ± √disc)/2
 *   • disc < 0               → complex pair (a rotation) — flagged, no real axis.
 * Eigenvectors solve (A − λI)v = 0. For a 2×2 row [a−λ, b] a null vector is
 * (b, λ−a) (or (λ−d, c) from the other row); we pick the better-conditioned one.
 */
export function eigen2(m: Mat2): EigenResult {
  const { a, b, c, d } = m;
  const trace = a + d;
  const det = a * d - b * c;
  const disc = trace * trace - 4 * det;

  if (disc < 0) {
    return { trace, det, disc, complex: true, pairs: [] };
  }

  const root = Math.sqrt(disc);
  const lambdas = [(trace + root) / 2, (trace - root) / 2];

  const pairs: EigenPair[] = lambdas.map((lambda) => {
    // (A − λI) = [[a−λ, b], [c, d−λ]]. Both rows are multiples of each other.
    // Null vector from row 1: (b, λ−a); from row 2: (λ−d, c). Use whichever
    // has the larger magnitude for numerical robustness.
    const cand1: Vec2 = { x: b, y: lambda - a };
    const cand2: Vec2 = { x: lambda - d, y: c };
    const pick = norm2(cand1) >= norm2(cand2) ? cand1 : cand2;
    let v = unit(pick);
    // Degenerate fallback: A = λI (scalar matrix) → every direction is an
    // eigenvector; pick a canonical axis so the two reported vectors differ.
    if (norm2(pick) < 1e-12) v = lambda === lambdas[0] ? { x: 1, y: 0 } : { x: 0, y: 1 };
    return { lambda, vec: v };
  });

  return { trace, det, disc, complex: false, pairs };
}

/** Eigen-decomposition of a SYMMETRIC 2×2 matrix [[p, q],[q, r]] (always real). */
function symEigen(p: number, q: number, r: number): { vals: [number, number]; vecs: [Vec2, Vec2] } {
  const tr = p + r;
  const dt = p * r - q * q;
  const disc = Math.max(0, tr * tr - 4 * dt); // ≥0 for symmetric matrices (clamp fp noise)
  const root = Math.sqrt(disc);
  const l1 = (tr + root) / 2; // larger
  const l2 = (tr - root) / 2; // smaller
  const eigVec = (lambda: number): Vec2 => {
    // null vector of [[p−λ, q],[q, r−λ]]
    const cand1: Vec2 = { x: q, y: lambda - p };
    const cand2: Vec2 = { x: lambda - r, y: q };
    const pick = norm2(cand1) >= norm2(cand2) ? cand1 : cand2;
    if (norm2(pick) < 1e-12) return lambda === l1 ? { x: 1, y: 0 } : { x: 0, y: 1 };
    return unit(pick);
  };
  return { vals: [l1, l2], vecs: [eigVec(l1), eigVec(l2)] };
}

export interface SvdResult {
  sigma1: number; sigma2: number;     // σ₁ ≥ σ₂ ≥ 0 (always real)
  v1: Vec2; v2: Vec2;                  // right singular vectors (columns of V)
  u1: Vec2; u2: Vec2;                  // left singular vectors (columns of U)
  cond: number;                        // κ = σ₁/σ₂ (∞ if σ₂≈0)
}

/**
 * SVD of a general 2×2 matrix A = U Σ Vᵀ.
 *   S = AᵀA is symmetric PSD; its eigenvalues are σᵢ² with eigenvectors = vᵢ
 *   (columns of V). σᵢ = √(eigenvalue). uᵢ = A·vᵢ / σᵢ (guarded for σ≈0); when
 *   σ₂≈0 we complete U with the perpendicular of u₁ to keep it orthonormal.
 */
export function svd2(m: Mat2): SvdResult {
  const { a, b, c, d } = m;
  // S = AᵀA = [[a²+c², ab+cd], [ab+cd, b²+d²]]
  const p = a * a + c * c;
  const q = a * b + c * d;
  const r = b * b + d * d;
  const { vals, vecs } = symEigen(p, q, r);

  const sigma1 = Math.sqrt(Math.max(0, vals[0]));
  const sigma2 = Math.sqrt(Math.max(0, vals[1]));
  const v1 = vecs[0];
  const v2 = vecs[1];

  const perp = (v: Vec2): Vec2 => ({ x: -v.y, y: v.x });
  const u1 = sigma1 > 1e-9 ? unit(apply(m, v1)) : { x: 1, y: 0 };
  const u2 = sigma2 > 1e-9 ? unit(apply(m, v2)) : perp(u1);

  const cond = sigma2 > 1e-9 ? sigma1 / sigma2 : Infinity;
  return { sigma1, sigma2, v1, v2, u1, u2, cond };
}

/** N points on the unit circle, in order (for drawing the circle → ellipse). */
export function unitCircle(n: number): Vec2[] {
  return Array.from({ length: n }, (_, i) => {
    const t = (i / n) * Math.PI * 2;
    return { x: Math.cos(t), y: Math.sin(t) };
  });
}

/** Angle (radians) of a vector, for SVG ellipse orientation. */
export const angleOf = (v: Vec2): number => Math.atan2(v.y, v.x);
