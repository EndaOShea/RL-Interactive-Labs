// Exact analytic 2×2 linear algebra for the Eigenvalues & SVD and Linear
// Transformations labs. All hand-rolled (no numeric solvers): the eigen-
// decomposition and SVD of a general 2×2 matrix have closed forms, so every
// number shown is exact to floating point.

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
/** Rotate a vector by +90°. */
export const perp = (v: Vec2): Vec2 => ({ x: -v.y, y: v.x });

export const det2 = (m: Mat2): number => m.a * m.d - m.b * m.c;
export const matMul = (p: Mat2, q: Mat2): Mat2 => ({
  a: p.a * q.a + p.b * q.c, b: p.a * q.b + p.b * q.d,
  c: p.c * q.a + p.d * q.c, d: p.c * q.b + p.d * q.d,
});
export const rot = (th: number): Mat2 => ({ a: Math.cos(th), b: -Math.sin(th), c: Math.sin(th), d: Math.cos(th) });
export const diag = (d1: number, d2: number): Mat2 => ({ a: d1, b: 0, c: 0, d: d2 });
/** (1−t)·P + t·Q. */
export const lerpMat = (p: Mat2, q: Mat2, t: number): Mat2 => ({
  a: (1 - t) * p.a + t * q.a, b: (1 - t) * p.b + t * q.b,
  c: (1 - t) * p.c + t * q.c, d: (1 - t) * p.d + t * q.d,
});
export const IDENTITY: Mat2 = { a: 1, b: 0, c: 0, d: 1 };

export interface EigenPair { lambda: number; vec: Vec2; } // vec is unit-length

/**
 *  distinct  — two different real eigenvalues, two eigen-directions
 *  scalar    — A = λI: one repeated λ and EVERY direction is an eigenvector
 *  defective — one repeated λ but only ONE eigen-direction (e.g. a shear)
 *  complex   — a conjugate pair α ± βi: no real invariant direction (A rotates)
 */
export type EigenKind = 'distinct' | 'scalar' | 'defective' | 'complex';

export interface EigenResult {
  trace: number;
  det: number;
  disc: number;            // t² − 4·det
  kind: EigenKind;
  complex: boolean;        // kind === 'complex'
  /** Real eigenpairs: 2 (distinct), 2 canonical axes (scalar), 1 (defective), 0 (complex). */
  pairs: EigenPair[];
  /** For a complex pair λ = re ± i·im: |λ| = √det and arg λ (radians, in (0, π)). */
  cplx: { re: number; im: number; modulus: number; angle: number } | null;
}

/**
 * Eigenvalues/eigenvectors of a 2×2 matrix via the characteristic equation
 * λ² − t·λ + det = 0, with t = trace, det = ad − bc.
 *   • disc = t² − 4·det > 0  → two real eigenvalues λ = (t ± √disc)/2
 *   • disc = 0               → one repeated λ = t/2 (scalar or defective)
 *   • disc < 0               → complex pair t/2 ± i·√(−disc)/2
 * Eigenvectors solve (A − λI)v = 0. For a 2×2 row [a−λ, b] a null vector is
 * (b, λ−a) (or (λ−d, c) from the other row); we pick the better-conditioned one.
 */
export function eigen2(m: Mat2): EigenResult {
  const { a, b, c, d } = m;
  const trace = a + d;
  const det = a * d - b * c;
  const disc = trace * trace - 4 * det;
  const scale = Math.max(1, trace * trace, 4 * Math.abs(det));
  const zeroTol = 1e-12 * scale;

  const nullVec = (lambda: number): Vec2 | null => {
    const cand1: Vec2 = { x: b, y: lambda - a };
    const cand2: Vec2 = { x: lambda - d, y: c };
    const pick = norm2(cand1) >= norm2(cand2) ? cand1 : cand2;
    return norm2(pick) < 1e-12 ? null : unit(pick);
  };

  if (disc < -zeroTol) {
    const re = trace / 2;
    const im = Math.sqrt(-disc) / 2;
    return {
      trace, det, disc, kind: 'complex', complex: true, pairs: [],
      cplx: { re, im, modulus: Math.hypot(re, im), angle: Math.atan2(im, re) },
    };
  }

  if (Math.abs(disc) <= zeroTol) {
    const lambda = trace / 2;
    const mtol = 1e-9 * Math.max(1, Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
    if (Math.abs(b) <= mtol && Math.abs(c) <= mtol && Math.abs(a - d) <= mtol) {
      // A = λI: every vector is an eigenvector — report the canonical axes (by index).
      return {
        trace, det, disc, kind: 'scalar', complex: false, cplx: null,
        pairs: [{ lambda, vec: { x: 1, y: 0 } }, { lambda, vec: { x: 0, y: 1 } }],
      };
    }
    const v = nullVec(lambda) ?? { x: 1, y: 0 };
    return { trace, det, disc, kind: 'defective', complex: false, cplx: null, pairs: [{ lambda, vec: v }] };
  }

  const root = Math.sqrt(disc);
  const lambdas = [(trace + root) / 2, (trace - root) / 2];
  const pairs: EigenPair[] = lambdas.map((lambda, i) => ({
    lambda,
    vec: nullVec(lambda) ?? (i === 0 ? { x: 1, y: 0 } : { x: 0, y: 1 }),
  }));
  return { trace, det, disc, kind: 'distinct', complex: false, cplx: null, pairs };
}

/** Largest eigenvalue and its unit eigenvector of a SYMMETRIC 2×2 [[p, q],[q, r]] (always real). */
function symTop(p: number, q: number, r: number): { l1: number; v1: Vec2 } {
  const tr = p + r;
  const dt = p * r - q * q;
  const root = Math.sqrt(Math.max(0, tr * tr - 4 * dt)); // ≥0 for symmetric matrices (clamp fp noise)
  const l1 = (tr + root) / 2;
  const cand1: Vec2 = { x: q, y: l1 - p };
  const cand2: Vec2 = { x: l1 - r, y: q };
  const pick = norm2(cand1) >= norm2(cand2) ? cand1 : cand2;
  // pick ≈ 0 only when the matrix is a multiple of I: then any direction is an eigenvector.
  return { l1, v1: norm2(pick) < 1e-12 ? { x: 1, y: 0 } : unit(pick) };
}

export interface SvdResult {
  sigma1: number; sigma2: number;     // σ₁ ≥ σ₂ ≥ 0 (always real)
  v1: Vec2; v2: Vec2;                  // right singular vectors (columns of V), v2 = perp(v1)
  u1: Vec2; u2: Vec2;                  // left singular vectors (columns of U), A·vᵢ = σᵢ·uᵢ
  cond: number;                        // κ = σ₁/σ₂ (∞ if σ₂≈0)
}

/**
 * SVD of a general 2×2 matrix A = U Σ Vᵀ.
 *   S = AᵀA is symmetric PSD; its top eigenvector is v₁ and v₂ = perp(v₁) (V is
 *   orthonormal by construction, even when σ₁ = σ₂). σ₁ = √λ_max(S), and
 *   σ₂ = |det A|/σ₁ (no cancellation). uᵢ = A·vᵢ/σᵢ; when σ₂ ≈ 0 U is completed
 *   with perp(u₁). So A·v₁ = σ₁u₁ and A·v₂ = σ₂u₂ hold exactly — including for
 *   reflections (det < 0), where U comes out with det U = −1.
 */
export function svd2(m: Mat2): SvdResult {
  const { a, b, c, d } = m;
  const p = a * a + c * c;
  const q = a * b + c * d;
  const r = b * b + d * d;
  const { l1, v1 } = symTop(p, q, r);
  const v2 = perp(v1);
  const sigma1 = Math.sqrt(Math.max(0, l1));
  const sigma2 = sigma1 > 1e-12 ? Math.abs(det2(m)) / sigma1 : 0;
  const u1 = sigma1 > 1e-12 ? unit(apply(m, v1)) : { x: 1, y: 0 };
  const u2 = sigma2 > 1e-12 * Math.max(1, sigma1) ? unit(apply(m, v2)) : perp(u1);
  const cond = sigma2 > 1e-12 * Math.max(1, sigma1) ? sigma1 / sigma2 : Infinity;
  return { sigma1, sigma2, v1, v2, u1, u2, cond };
}

/**
 * A rotation–scale–rotation factorisation for animation: A = R(φu)·diag(σ₁, s₂)·R(−φv),
 * with both outer factors PROPER rotations. s₂ = det A/σ₁ = ±σ₂ carries the sign of
 * det A, so a reflection shows up as the second axis being scaled through 0 to −σ₂.
 */
export interface SvdStages { phiV: number; phiU: number; s1: number; s2: number; }

export function svdStages(m: Mat2): SvdStages {
  const svd = svd2(m);
  const s1 = svd.sigma1;
  const s2 = s1 > 1e-12 ? det2(m) / s1 : 0;
  // (v₁, u₁) and (−v₁, −u₁) are equally valid — take the one with less total rotation.
  const option = (sgn: 1 | -1) => {
    const v: Vec2 = { x: sgn * svd.v1.x, y: sgn * svd.v1.y };
    const u = s1 > 1e-12 ? unit(apply(m, v)) : { x: 1, y: 0 };
    const phiV = Math.atan2(v.y, v.x);
    const phiU = Math.atan2(u.y, u.x);
    return { phiV, phiU, cost: Math.abs(phiV) + Math.abs(phiU) };
  };
  const o1 = option(1), o2 = option(-1);
  const o = o2.cost < o1.cost ? o2 : o1;
  return { phiV: o.phiV, phiU: o.phiU, s1, s2 };
}

export type StageName = 'Vᵀ rotate' | 'Σ stretch' | 'U rotate';
export const STAGE_T_MAX = 3;

/**
 * The frame matrix at animation time t ∈ [0, 3]:
 *   [0,1] R(−φv·t)                       — Vᵀ turns the right-singular axes onto x, y
 *   [1,2] diag(1+(σ₁−1)τ, 1+(s₂−1)τ)·Vᵀ  — Σ stretches along the axes
 *   [2,3] R(φu·τ)·Σ·Vᵀ                    — U turns the stretched axes onto u₁, u₂
 * At t = 3 it equals A exactly (up to rounding).
 */
export function stageMatrix(st: SvdStages, t: number): Mat2 {
  const tt = Math.max(0, Math.min(STAGE_T_MAX, t));
  const Vt = rot(-st.phiV);
  if (tt <= 1) return rot(-st.phiV * tt);
  if (tt <= 2) {
    const tau = tt - 1;
    return matMul(diag(1 + (st.s1 - 1) * tau, 1 + (st.s2 - 1) * tau), Vt);
  }
  const tau = tt - 2;
  return matMul(rot(st.phiU * tau), matMul(diag(st.s1, st.s2), Vt));
}

export const stageOf = (t: number): StageName => (t <= 1 ? 'Vᵀ rotate' : t <= 2 ? 'Σ stretch' : 'U rotate');

/** N points on the unit circle, in order (for drawing the circle → ellipse). */
export function unitCircle(n: number): Vec2[] {
  return Array.from({ length: n }, (_, i) => {
    const t = (i / n) * Math.PI * 2;
    return { x: Math.cos(t), y: Math.sin(t) };
  });
}

/** Angle (radians) of a vector, for ellipse orientation. */
export const angleOf = (v: Vec2): number => Math.atan2(v.y, v.x);

/** Display text for the eigenvalues of A, by kind. */
export function eigenText(e: EigenResult, fmt: (v: number) => string): string {
  if (e.kind === 'complex' && e.cplx) {
    return `${fmt(e.cplx.re)} ± ${fmt(e.cplx.im)}i`;
  }
  if (e.kind === 'scalar') return `${fmt(e.pairs[0]?.lambda ?? NaN)} (×2, every direction)`;
  if (e.kind === 'defective') return `${fmt(e.pairs[0]?.lambda ?? NaN)} (×2, one direction)`;
  return e.pairs.map((p) => fmt(p.lambda)).join(', ');
}
