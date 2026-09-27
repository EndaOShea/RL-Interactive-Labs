// Runnable Python exports for the Maths labs (template strings — not LLM
// generated). Each script mirrors its lab exactly: the same functions, the
// parameters read from the live lab state at download time, the same update
// rules, tolerances and checks.
import type { PythonSample } from '../../utils/pythonSamples';
import type { Vec2, Mat2 } from './matrix-multiplication';
import type { GdFn, GdOpt } from './gradient-descent';
import type { TaylorFn } from './taylor-series';
import type { DerivFnId, DiffMethod } from './finite-difference';
import {
  GD_FNS, GD_RHO, GD_B1, GD_B2, GD_EPS, GD_TOL, GD_DIVERGE_MARGIN, GD_FD_H,
} from './gradient-descent';
import { TAYLOR_FNS, TAYLOR_MAX_CAP, PADE_SINGULAR_TOL } from './taylor-series';
import { DERIV_FNS, DX_LOG_MIN, DX_LOG_MAX, DX_CURVE_N } from './finite-difference';
import { CHAIN_PRESETS } from './chain-rule';
import {
  SURFACES, CONVERGE_TOL, BASIN_MAX_STEPS, CURVE_N, scatterStartsFor,
} from './convex-optimization';

/** A JS number as a Python float literal. */
const num = (n: number): string => {
  const s = String(n);
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

/** Plain-ASCII version of a display label (so every script prints on any console). */
const ascii = (s: string): string => s
  .replace(/²/g, '^2').replace(/³/g, '^3').replace(/⁴/g, '^4').replace(/ˣ/g, '^x').replace(/ᵘ/g, '^u')
  .replace(/[−–]/g, '-').replace(/·/g, '*').replace(/∂/g, 'd').replace(/π/g, 'pi')
  .replace(/±/g, '+-').replace(/√/g, 'sqrt').replace(/[^\x20-\x7e]/g, '?');

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------
export const gradientDescentPython = (
  fn: GdFn,
  lr: number,
  opt: GdOpt = 'momentum',
  beta = 0.7,
  startX = 1.3,
) => {
  const d = GD_FNS[fn];
  const [lo, hi] = d.domain;
  const update = {
    momentum: `        # Heavy-ball momentum: v <- beta*v - alpha*g ; x <- x + v   (beta = 0 is plain GD)
        v = BETA * v - ALPHA * g
        dx = v`,
    rmsprop: `        # RMSProp: s <- rho*s + (1-rho)*g^2 ; x <- x - alpha*g / (sqrt(s) + eps)
        s = RHO * s + (1 - RHO) * g * g
        dx = -(ALPHA * g) / (np.sqrt(s) + EPS)
        v = dx`,
    adam: `        # Adam: bias-corrected first (m) and second (s) moments
        m = B1 * m + (1 - B1) * g
        s = B2 * s + (1 - B2) * g * g
        mhat = m / (1 - B1 ** (t + 1))
        shat = s / (1 - B2 ** (t + 1))
        dx = -(ALPHA * mhat) / (np.sqrt(shat) + EPS)
        v = dx`,
    newton: `        # Newton: jump to the stationary point of the local parabola, x <- x - f'/f''
        h = d2f(x)
        dx = -g / h if abs(h) > 1e-8 else -ALPHA * g
        v = dx`,
  }[opt];
  const convTest = opt === 'newton'
    ? 'abs(g_next) < TOL'
    : 'abs(g_next) < TOL and abs(dx) < TOL';

  return `import numpy as np

# Gradient descent -- mirrors the lab: f(x) = ${ascii(d.formula.replace('f(x)=', ''))}, optimiser = ${opt}.
# Same update rule, constants and stop tests as the on-screen run.
ALPHA = ${num(lr)}          # learning rate (Newton: fallback step when |f''| <= 1e-8)
BETA = ${num(beta)}         # heavy-ball momentum (momentum optimiser only; 0 = plain GD)
X0 = ${num(startX)}         # start position
RHO = ${num(GD_RHO)}            # RMSProp decay
B1, B2 = ${num(GD_B1)}, ${num(GD_B2)}  # Adam moment decays
EPS = ${GD_EPS}
TOL = ${GD_TOL}           # stop when |f'(x')| < TOL (first-order methods: and |dx| < TOL)
LO, HI = ${num(lo)}, ${num(hi)}      # plotted domain
DIVERGE_MARGIN = ${num(GD_DIVERGE_MARGIN)}  # x more than this far outside the domain => diverged
MAX_STEPS = 10000       # the lab keeps stepping until it stops; this caps the script
FD_H = ${GD_FD_H}           # central-difference gradient check step


def f(x):
    return ${d.py.f}


def df(x):
    return ${d.py.df}       # analytic 1st derivative


def d2f(x):
    return ${d.py.d2f}      # analytic 2nd derivative (curvature)


def fd_grad(x, h=FD_H):
    # independent check of df: central difference
    return (f(x + h) - f(x - h)) / (2 * h)


def alpha_limit(x, beta=BETA):
    # heavy-ball GD on the local quadratic model stays bounded iff alpha * f'' < 2 (1 + beta)
    h = d2f(x)
    return 2 * (1 + beta) / h if h > 0 else float("inf")


def descend(x0):
    x, v, m, s = float(x0), 0.0, 0.0, 0.0
    trail = [x]
    for t in range(MAX_STEPS):
        g = df(x)
${update}
        nx = x + dx
        diverged = (not np.isfinite(nx)) or nx < LO - DIVERGE_MARGIN or nx > HI + DIVERGE_MARGIN
        g_next = df(nx)
        converged = (not diverged) and (${convTest})
        x = nx
        trail.append(x)
        if diverged:
            return x, t + 1, "diverged", trail
        if converged:
            return x, t + 1, "converged", trail
    return x, MAX_STEPS, "still moving", trail


if __name__ == "__main__":
    print(f"optimiser = ${opt}   alpha = {ALPHA}   beta = {BETA}   x0 = {X0}")
    print(f"at x0: f' = {df(X0):.6f} (central difference {fd_grad(X0):.6f}), f'' = {d2f(X0):.4f}")${opt === 'momentum' ? `
    print(f"local stability limit 2(1+beta)/f''(x0) = {alpha_limit(X0):.4f}  (alpha = {ALPHA})")` : ''}
    x_end, steps, outcome, trail = descend(X0)
    print("first steps:", " -> ".join(f"{p:.3f}" for p in trail[:12]))
    print(f"{outcome} after {steps} steps: x = {x_end:.6g}, f(x) = {f(x_end):.6g}, f'(x) = {df(x_end):.2e}")
    if outcome == "converged":
        kind = "a minimum (f'' > 0)" if d2f(x_end) > 0 else "a maximum/saddle (f'' <= 0)"
        print("the stationary point is", kind)
`;
};

// ---------------------------------------------------------------------------
// Taylor & Padé
// ---------------------------------------------------------------------------
const TAYLOR_PY: Record<TaylorFn, { f: string; coef: string; radius: string }> = {
  sin: {
    f: 'np.sin(x)',
    coef: `    d = [np.sin(a), np.cos(a), -np.sin(a), -np.cos(a)]
    return [d[k % 4] / factorial(k) for k in range(deg + 1)]`,
    radius: 'float("inf")',
  },
  cos: {
    f: 'np.cos(x)',
    coef: `    d = [np.cos(a), -np.sin(a), -np.cos(a), np.sin(a)]
    return [d[k % 4] / factorial(k) for k in range(deg + 1)]`,
    radius: 'float("inf")',
  },
  exp: {
    f: 'np.exp(x)',
    coef: `    ea = np.exp(a)
    return [ea / factorial(k) for k in range(deg + 1)]`,
    radius: 'float("inf")',
  },
  geom: {
    f: '1.0 / (1.0 - x)',
    coef: '    return [1.0 / (1.0 - a) ** (k + 1) for k in range(deg + 1)]',
    radius: 'abs(1.0 - a)',
  },
  log: {
    f: 'np.log(1.0 + x)',
    coef: '    return [np.log(1.0 + a)] + [(-1.0) ** (k - 1) / (k * (1.0 + a) ** k) for k in range(1, deg + 1)]',
    radius: 'abs(1.0 + a)',
  },
  tanh: {
    f: 'np.tanh(x)',
    coef: `    # exact recurrence from tanh' = 1 - tanh^2:  (k+1) c_{k+1} = [k == 0] - sum_{i+j=k} c_i c_j
    c = [np.tanh(a)]
    for k in range(deg):
        conv = 0.0
        for i in range(k + 1):
            conv += c[i] * c[k - i]
        c.append(((1.0 if k == 0 else 0.0) - conv) / (k + 1))
    return c`,
    radius: 'float(np.hypot(a, np.pi / 2))',
  },
  runge: {
    f: '1.0 / (1.0 + 25.0 * x * x)',
    coef: `    # exact recurrence from (1 + 25x^2) f = 1 with x = a + t:
    #   c_k = -(50 a c_{k-1} + 25 c_{k-2}) / (1 + 25 a^2)
    D = 1 + 25 * a * a
    c = [1.0 / D]
    for k in range(1, deg + 1):
        prev2 = c[k - 2] if k >= 2 else 0.0
        c.append(-(50 * a * c[k - 1] + 25 * prev2) / D)
    return c`,
    radius: 'float(np.hypot(a, 0.2))',
  },
};

export const taylorPython = (
  fn: TaylorFn,
  mode: 'taylor' | 'pade' = 'taylor',
  a = 0,
  evalX = 2,
  maxDeg = 8,
) => {
  const def = TAYLOR_FNS[fn];
  const p = TAYLOR_PY[fn];
  return `import numpy as np
from math import factorial

# Taylor / Pade approximation -- mirrors the lab (f(x) = ${ascii(def.label)}, mode = ${mode}).
# Exact coefficients (closed form or exact recurrence -- no numerical derivatives),
# the radius of convergence about A, and the lab's [m/m] Pade solver.
MODE = "${mode}"
A = ${num(a)}              # expansion centre
EVAL = ${num(evalX)}           # where the error is measured
MAX_DEG = ${Math.min(TAYLOR_MAX_CAP, Math.max(1, Math.round(maxDeg)))}           # Run grows the degree n = 0 .. MAX_DEG
COEF_DEG = ${TAYLOR_MAX_CAP}          # coefficients are computed up to the lab's cap
PADE_SINGULAR_TOL = ${PADE_SINGULAR_TOL}


def f(x):
    return ${p.f}


def coeffs(a, deg):
    # c_k = f^(k)(a) / k!  for k = 0..deg
${p.coef}


def radius(a):
    # distance from a to the nearest singularity: ${ascii(def.singularity)}
    return ${p.radius}


def radius_case(R, dist):
    if not np.isfinite(R):
        return "inside"
    if abs(dist - R) <= 1e-9:
        return "boundary"
    return "inside" if dist < R else "outside"


def poly_eval(c, t):
    # Horner's rule
    s = 0.0
    for ck in reversed(c):
        s = s * t + ck
    return s


def taylor(c, a, n, x):
    return poly_eval(c[: n + 1], x - a)


def solve_linear(M, rhs, tol=PADE_SINGULAR_TOL):
    # Gaussian elimination with partial pivoting; None when a pivot <= tol * max|M|
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(M)]
    scale = max(abs(v) for row in M for v in row)
    if scale == 0:
        return None
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[piv][col]):
                piv = r
        if abs(aug[piv][col]) <= tol * scale:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            fct = aug[r][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[r][k] = aug[r][k] - fct * aug[col][k]
    z = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for k in range(i + 1, n):
            s -= aug[i][k] * z[k]
        z[i] = s / aug[i][i]
    return z


def pade(c, requested):
    # [m/m] Pade from c[0..2m]: Q (q0 = 1) solves sum_j q_j c_{m+i-j} = 0 (i = 1..m),
    # then p_i = sum_k c_{i-k} q_k. While the Toeplitz system is singular, lower m.
    for m in range(min(requested, (len(c) - 1) // 2), 0, -1):
        M = [[c[m + i - j] for j in range(m)] for i in range(m)]
        rhs = [-c[m + i + 1] for i in range(m)]
        b = solve_linear(M, rhs)
        if b is None:
            continue
        q = [1.0] + b
        p = [sum(c[i - k] * q[k] for k in range(min(i, m) + 1)) for i in range(m + 1)]
        return p, q, m
    return [c[0]], [1.0], 0


def pade_eval(p, q, a, x):
    t = x - a
    den = poly_eval(q, t)
    mag = sum(abs(qk) * abs(t) ** k for k, qk in enumerate(q))
    if abs(den) <= 1e-12 * mag:
        return float("nan")      # a pole of the approximant
    return poly_eval(p, t) / den


if __name__ == "__main__":
    c = coeffs(A, COEF_DEG)
    R = radius(A)
    dist = abs(EVAL - A)
    fx = f(EVAL)
    print(f"f(x) = ${ascii(def.label)}   centre a = {A}   eval x = {EVAL}   f(x) = {fx:.10g}")
    print(f"radius of convergence R = {R:.4f}; |x - a| = {dist:.4f} -> {radius_case(R, dist)}")
    print("c_k:", ", ".join(f"{v:.6g}" for v in c[: MAX_DEG + 1]))
    print()
    for n in range(MAX_DEG + 1):
        T = taylor(c, A, n, EVAL)
        row = f"n={n:2d}  T_n={T:+.10g}  |err|={abs(fx - T):.3e}"
        if MODE == "pade":
            m = n // 2
            p, q, used = pade(c[: 2 * m + 1], m)
            Rv = pade_eval(p, q, A, EVAL)
            note = "" if used == m else f" (the [{m}/{m}] system is singular)"
            row += f"   Pade[{used}/{used}]{note}={Rv:+.10g}  |err|={abs(fx - Rv):.3e}"
        print(row)
`;
};

// ---------------------------------------------------------------------------
// Shared closed-form 2x2 eigen / SVD (Linear Transformations + Eigenvalues & SVD)
// ---------------------------------------------------------------------------
const EIGEN_SVD_CORE = `def apply(M, v):
    return np.array([M[0, 0] * v[0] + M[0, 1] * v[1], M[1, 0] * v[0] + M[1, 1] * v[1]])


def unit(v):
    n = np.hypot(v[0], v[1])
    return np.array([0.0, 0.0]) if n < 1e-12 else v / n


def perp(v):
    return np.array([-v[1], v[0]])


def eigen2(M):
    # characteristic equation lambda^2 - t*lambda + det = 0, disc = t^2 - 4 det
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    zero_tol = 1e-12 * max(1.0, tr * tr, 4 * abs(det))

    def null_vec(lam):
        cand1 = np.array([b, lam - a])
        cand2 = np.array([lam - d, c])
        pick = cand1 if np.hypot(*cand1) >= np.hypot(*cand2) else cand2
        return None if np.hypot(*pick) < 1e-12 else unit(pick)

    out = dict(trace=tr, det=det, disc=disc)
    if disc < -zero_tol:
        re, im = tr / 2, np.sqrt(-disc) / 2
        out.update(kind="complex", pairs=[], re=re, im=im, modulus=np.hypot(re, im), angle=np.arctan2(im, re))
        return out
    if abs(disc) <= zero_tol:
        lam = tr / 2
        mtol = 1e-9 * max(1.0, abs(a), abs(b), abs(c), abs(d))
        if abs(b) <= mtol and abs(c) <= mtol and abs(a - d) <= mtol:
            # A = lambda I: every direction is an eigenvector (report the axes)
            out.update(kind="scalar", pairs=[(lam, np.array([1.0, 0.0])), (lam, np.array([0.0, 1.0]))])
            return out
        v = null_vec(lam)
        out.update(kind="defective", pairs=[(lam, v if v is not None else np.array([1.0, 0.0]))])
        return out
    root = np.sqrt(disc)
    pairs = []
    for i, lam in enumerate([(tr + root) / 2, (tr - root) / 2]):
        v = null_vec(lam)
        if v is None:
            v = np.array([1.0, 0.0]) if i == 0 else np.array([0.0, 1.0])
        pairs.append((lam, v))
    out.update(kind="distinct", pairs=pairs)
    return out


def svd2(M):
    # v1 = top eigenvector of M^T M, v2 = perp(v1); s1 = sqrt(lambda_max), s2 = |det| / s1;
    # u_i = M v_i / s_i (perp(u1) when s2 ~ 0)
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    p, q, r = a * a + c * c, a * b + c * d, b * b + d * d
    tr, dt = p + r, p * r - q * q
    root = np.sqrt(max(0.0, tr * tr - 4 * dt))
    l1 = (tr + root) / 2
    cand1 = np.array([q, l1 - p])
    cand2 = np.array([l1 - r, q])
    pick = cand1 if np.hypot(*cand1) >= np.hypot(*cand2) else cand2
    v1 = np.array([1.0, 0.0]) if np.hypot(*pick) < 1e-12 else unit(pick)
    v2 = perp(v1)
    s1 = np.sqrt(max(0.0, l1))
    s2 = abs(a * d - b * c) / s1 if s1 > 1e-12 else 0.0
    u1 = unit(apply(M, v1)) if s1 > 1e-12 else np.array([1.0, 0.0])
    u2 = unit(apply(M, v2)) if s2 > 1e-12 * max(1.0, s1) else perp(u1)
    cond = s1 / s2 if s2 > 1e-12 * max(1.0, s1) else float("inf")
    return s1, s2, v1, v2, u1, u2, cond


def svd_stages(M):
    # M = R(phi_u) diag(s1, s2_signed) R(-phi_v) with proper rotations; s2_signed = det/s1
    s1, _, v1, _, _, _, _ = svd2(M)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    s2 = det / s1 if s1 > 1e-12 else 0.0
    best = None
    for sgn in (1.0, -1.0):
        v = sgn * v1
        u = unit(apply(M, v)) if s1 > 1e-12 else np.array([1.0, 0.0])
        phi_v, phi_u = np.arctan2(v[1], v[0]), np.arctan2(u[1], u[0])
        cost = abs(phi_v) + abs(phi_u)
        if best is None or cost < best[2]:
            best = (phi_v, phi_u, cost)
    return best[0], best[1], s1, s2


def rot(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def stage_matrix(st, t):
    # t in [0,1]: V^T turns;  [1,2]: Sigma stretches;  [2,3]: U turns.  stage_matrix(st, 3) == M
    phi_v, phi_u, s1, s2 = st
    t = min(max(t, 0.0), 3.0)
    Vt = rot(-phi_v)
    if t <= 1:
        return rot(-phi_v * t)
    if t <= 2:
        tau = t - 1
        return np.diag([1 + (s1 - 1) * tau, 1 + (s2 - 1) * tau]) @ Vt
    tau = t - 2
    return rot(phi_u * tau) @ (np.diag([s1, s2]) @ Vt)


def describe_eigen(e):
    if e["kind"] == "complex":
        return (f"complex pair {e['re']:.4f} +- {e['im']:.4f}i, |lambda| = sqrt(det) = {e['modulus']:.4f}, "
                f"arg = {np.degrees(e['angle']):.2f} deg (no real eigenvector)")
    if e["kind"] == "scalar":
        return f"lambda = {e['pairs'][0][0]:.4f} repeated and A = lambda*I: every direction is an eigenvector"
    if e["kind"] == "defective":
        lam, v = e["pairs"][0]
        return f"repeated lambda = {lam:.4f} with ONE eigen-direction {np.round(v, 4)} (defective)"
    return "; ".join(f"lambda = {lam:.4f}, v = {np.round(v, 4)}" for lam, v in e["pairs"])
`;

export const linearTransformPython = (
  mode: 'eigen' | 'svd' = 'eigen',
  a = 1, b = 1, c = 0, d = 1,
) => `import numpy as np

# Linear transformations of the plane -- mirrors the lab (mode = ${mode}).
# Columns of M are where the basis vectors i-hat, j-hat land. Closed-form 2x2
# eigen / SVD exactly as on screen, cross-checked against numpy.
MODE = "${mode}"
M = np.array([[${num(a)}, ${num(b)}],
              [${num(c)}, ${num(d)}]])

${EIGEN_SVD_CORE}

if __name__ == "__main__":
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    print("M =\\n", M)
    print(f"det = {det:.4f}  (area scale; < 0 flips orientation)")
    e = eigen2(M)
    print("eigen:", describe_eigen(e))
    print("numpy eigvals:", np.round(np.linalg.eigvals(M), 4))

    s1, s2, v1, v2, u1, u2, cond = svd2(M)
    U, S, Vt = np.column_stack([u1, u2]), np.diag([s1, s2]), np.vstack([v1, v2])
    print(f"\\nsingular values {s1:.4f}, {s2:.4f}  (numpy: {np.round(np.linalg.svd(M, compute_uv=False), 4)})")
    print("condition number =", "inf" if not np.isfinite(cond) else round(float(cond), 4))
    print("|M - U S V^T| =", float(np.abs(M - U @ S @ Vt).max()))

    if MODE == "eigen":
        # Run: M(t) = (1-t) I + t M has M's eigenvectors, with eigenvalues (1-t) + t*lambda
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            F = (1 - t) * np.eye(2) + t * M
            checks = [float(np.abs(F @ v - ((1 - t) + t * lam) * v).max()) for lam, v in e["pairs"]]
            print(f"t={t:.2f}  det(M(t)) = {np.linalg.det(F):+.4f}  max|M(t) v - ((1-t)+t lambda) v| = {max(checks, default=0.0):.1e}")
    else:
        st = svd_stages(M)
        print(f"\\nRun: V^T rotates by {np.degrees(-st[0]):.2f} deg, Sigma scales by {st[2]:.4f} and {st[3]:.4f}, U rotates by {np.degrees(st[1]):.2f} deg")
        for t in (0.0, 1.0, 1.5, 2.0, 3.0):
            F = stage_matrix(st, t)
            print(f"t={t:.1f}  det(frame) = {np.linalg.det(F):+.4f}  frame = {np.round(F, 4).tolist()}")
        print("final frame == M:", float(np.abs(stage_matrix(st, 3.0) - M).max()) < 1e-12)
`;

export const eigenSvdPython = (
  a: number,
  b: number,
  c: number,
  d: number,
) => `import numpy as np

# Eigenvalues & SVD of a 2x2 matrix -- mirrors the lab. Closed forms exactly as on
# screen: A maps the unit circle to an ellipse; SVD reads A = U diag(s) V^T as
# rotate (V^T) -> scale (s) -> rotate (U), which the lab's Run animates.
A = np.array([[${num(a)}, ${num(b)}],
              [${num(c)}, ${num(d)}]])

${EIGEN_SVD_CORE}

if __name__ == "__main__":
    e = eigen2(A)
    print(f"trace = {e['trace']:.4f}   det = {e['det']:.4f}   disc = t^2 - 4 det = {e['disc']:.4f}")
    print("eigen:", describe_eigen(e))
    print("numpy eigvals:", np.round(np.linalg.eigvals(A), 4))

    s1, s2, v1, v2, u1, u2, cond = svd2(A)
    U, S, Vt = np.column_stack([u1, u2]), np.diag([s1, s2]), np.vstack([v1, v2])
    print(f"\\nsingular values s = {s1:.4f}, {s2:.4f}  (numpy: {np.round(np.linalg.svd(A, compute_uv=False), 4)})")
    print("condition number =", "inf" if not np.isfinite(cond) else round(float(cond), 4))
    print("V =\\n", np.round(np.column_stack([v1, v2]), 4))
    print("U =\\n", np.round(U, 4))
    print("reconstruction error |A - U S V^T| =", float(np.abs(A - U @ S @ Vt).max()))

    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ellipse = A @ np.stack([np.cos(theta), np.sin(theta)])
    radii = np.hypot(ellipse[0], ellipse[1])
    print(f"unit circle -> ellipse: longest radius {radii.max():.4f} (s1), shortest {radii.min():.4f} (s2)")

    st = svd_stages(A)
    print(f"\\nRun: V^T rotates by {np.degrees(-st[0]):.2f} deg, Sigma scales by {st[2]:.4f} and {st[3]:.4f}"
          f"{' (negative: det < 0, a reflection)' if st[3] < 0 else ''}, U rotates by {np.degrees(st[1]):.2f} deg")
    for t in (0.0, 1.0, 1.5, 2.0, 3.0):
        print(f"t={t:.1f}  det(frame) = {np.linalg.det(stage_matrix(st, t)):+.4f}")
    print("final frame == A:", float(np.abs(stage_matrix(st, 3.0) - A).max()) < 1e-12)
`;

// ---------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------
export const derivativesPython = (
  fnId: DerivFnId = 'square',
  x0 = 1.0,
  method: DiffMethod = 'forward',
  dx = 1.0,
) => {
  const d = DERIV_FNS.find((f) => f.id === fnId) ?? DERIV_FNS[0]!;
  return `import numpy as np

# Derivative as the limit of a difference quotient -- mirrors the lab.
#   forward: [f(x+dx) - f(x)] / dx          error ~ f''(x) dx / 2   (first order)
#   central: [f(x+dx) - f(x-dx)] / (2 dx)   error ~ f'''(x) dx^2 / 6 (second order)
# In double precision the subtraction also loses digits as dx shrinks (round-off
# ~ eps |f| / dx), so the error bottoms out at an intermediate dx.
FN_NAME = "${d.py.name}"
X0 = ${num(x0)}
METHOD = "${method}"
DX = ${num(dx)}                   # the lab's current dx
LOG_MIN, LOG_MAX, N_CURVE = ${num(DX_LOG_MIN)}, ${num(DX_LOG_MAX)}, ${DX_CURVE_N}   # the lab's error-curve grid


def f(x):
    return ${d.py.f}


def df(x):
    return ${d.py.df}      # analytic derivative (closed form)


def quotient(x0, dx, method):
    if method == "forward":
        return (f(x0 + dx) - f(x0)) / dx
    return (f(x0 + dx) - f(x0 - dx)) / (2 * dx)


def dx_of(log_dx):
    # 10**log_dx rounded to 6 significant digits, as the lab does
    return float("%.6g" % (10.0 ** log_dx))


def error_curve(x0, method):
    exact = df(x0)
    out = []
    for i in range(N_CURVE):
        dxi = dx_of(LOG_MIN + (i / (N_CURVE - 1)) * (LOG_MAX - LOG_MIN))
        out.append((dxi, abs(quotient(x0, dxi, method) - exact)))
    return out


if __name__ == "__main__":
    exact = df(X0)
    print(f"f(x) = {FN_NAME}   x0 = {X0}   f'(x0) = {exact:.12g} (analytic)")
    for m in ("forward", "central"):
        q = quotient(X0, DX, m)
        mark = "  <- lab setting" if m == METHOD else ""
        print(f"{m:>8} at dx = {DX:.6g}: {q:.12g}   |error| = {abs(q - exact):.3e}{mark}")
    print()
    print(f"{'dx':>8}  {'forward |err|':>14}  {'central |err|':>14}")
    for k in range(0, 13):
        dxk = dx_of(-k)
        ef = abs(quotient(X0, dxk, "forward") - exact)
        ec = abs(quotient(X0, dxk, "central") - exact)
        print(f"{dxk:>8.0e}  {ef:>14.3e}  {ec:>14.3e}")
    print()
    for m in ("forward", "central"):
        curve = error_curve(X0, m)
        best = min(curve, key=lambda pt: pt[1])
        e1 = abs(quotient(X0, 1e-2, m) - exact)
        e10 = abs(quotient(X0, 1e-1, m) - exact)
        if e10 > 1e-12 and e1 > 0:
            order = f"{np.log10(e10 / e1):.3f}"
        else:
            order = "n/a (no truncation error left: only round-off)"
        print(f"{m:>8}: smallest error on the {N_CURVE}-point curve = {best[1]:.3e} at dx = {best[0]:.3g};"
              f" observed order p (dx 1e-2 vs 1e-1) = {order}")
`;
};

// ---------------------------------------------------------------------------
// Chain rule
// ---------------------------------------------------------------------------
export const chainRulePython = (
  presetId = 'sin_sq',
  x0 = 1.2,
) => {
  const p = CHAIN_PRESETS.find((q) => q.id === presetId) ?? CHAIN_PRESETS[0]!;
  const inputsOf = (name: string) => p.nodes.find((n) => n.name === name)?.inputs ?? [];
  const nodes = p.nodes
    .map((n) => `    ("${n.name}", (${n.inputs.map((k) => `"${k}"`).join(', ')},), lambda ${n.inputs.join(', ')}: ${n.py}, "${ascii(n.expr)}"),`)
    .join('\n');
  const edges = p.edges
    .map((e) => `    ("${e.from}", "${e.to}", lambda ${inputsOf(e.to).join(', ')}: ${e.py}, "${ascii(e.label)}"),`)
    .join('\n');
  return `import numpy as np

# Chain rule -- mirrors the lab (composite ${ascii(p.formula)}).
# dy/dx = SUM over every path x -> ... -> y of the PRODUCT of the local partials
# along it (one path for a plain chain). Backprop (reverse mode) reaches the same
# number by adding adjoints where a value fans out; a central finite difference of
# the whole composite checks it independently.
X0 = ${num(x0)}


def composite(x):
    return ${p.compositePy}


# nodes in topological order: (name, inputs, value(*inputs), expression)
NODES = [
${nodes}
]
# edges: (from, to, d(to)/d(from) evaluated at the TO node's inputs, label)
EDGES = [
${edges}
]
INPUTS = {name: inputs for name, inputs, _, _ in NODES}
OUT = NODES[-1][0]


def forward(x0):
    val = {"x": x0}
    for name, inputs, fn, expr in NODES:
        val[name] = fn(*[val[k] for k in inputs])
        print(f"forward:  {expr:<14} -> {name} = {val[name]:.6f}")
    return val


def local_partials(val):
    return [fn(*[val[k] for k in INPUTS[to]]) for frm, to, fn, label in EDGES]


def all_paths(at="x", acc=()):
    if at == OUT:
        return [list(acc)]
    paths = []
    for i, (frm, to, _, _) in enumerate(EDGES):
        if frm == at:
            paths += all_paths(to, acc + (i,))
    return paths


def backprop(local):
    adj = {OUT: 1.0}
    for name, _, _, _ in reversed(NODES):
        a = adj.get(name, 0.0)
        for i, (frm, to, _, _) in enumerate(EDGES):
            if to == name:
                adj[frm] = adj.get(frm, 0.0) + a * local[i]
    return adj


def finite_diff(x0, h=1e-4):
    return (composite(x0 + h) - composite(x0 - h)) / (2 * h)


if __name__ == "__main__":
    print(f"x = {X0}")
    val = forward(X0)
    local = local_partials(val)
    for (frm, to, _, label), v in zip(EDGES, local):
        print(f"local:    {label:<18} = {v:.6f}")
    paths = sorted(all_paths(), key=len)
    total = 0.0
    for path in paths:
        prod = 1.0
        for i in path:
            prod *= local[i]
        total += prod
        route = " -> ".join(["x"] + [EDGES[i][1] for i in path])
        factors = " * ".join(f"{local[i]:.4f}" for i in reversed(path))
        print(f"path {route:<16} product = {factors} = {prod:.6f}")
    print(f"\\ndy/dx = sum over {len(paths)} path(s) = {total:.6f}")
    print(f"backprop adjoint dy/dx = {backprop(local)['x']:.6f}")
    fd = finite_diff(X0)
    print(f"finite-difference check  = {fd:.6f}   agree: {abs(total - fd) < 1e-2}")
`;
};

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------
export const matmulPython = (
  a: Vec2,
  b: Vec2,
  A: Mat2,
  B: Mat2 = [1.5, 0, 0, 0.5],
) => `import numpy as np

# Matrix multiplication -- mirrors the lab.
# Part A: the dot product (a.b)          -- the operation every neuron computes.
# Part B: matrix . vector (y = A @ x)    -- the operation every dense layer performs.
# Part C: matrix . matrix (composition)  -- B(Ax) = (BA)x, and AB != BA in general.

# --- vectors (Part A) -------------------------------------------------------
a = np.array([${num(a[0])}, ${num(a[1])}])
b = np.array([${num(b[0])}, ${num(b[1])}])

dot = a @ b                       # = a[0]*b[0] + a[1]*b[1]
na, nb = np.linalg.norm(a), np.linalg.norm(b)
cos_t = dot / (na * nb) if na * nb > 1e-9 else 0.0   # a.b = |a||b| cos(theta)
theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))

# projection of a onto b
scalar_proj = dot / nb if nb > 1e-9 else 0.0          # signed length of a's shadow on b
vector_proj = (dot / (b @ b)) * b if b @ b > 1e-9 else np.zeros(2)

print("a . b        =", dot, "= a1*b1 + a2*b2 =", a[0]*b[0], "+", a[1]*b[1])
print("|a||b|cos(t) =", round(float(na * nb * cos_t), 4), " (same number, geometric form)")
print("cos(theta)   =", round(float(cos_t), 4), " theta =", round(float(theta), 1), "deg")
print("scalar proj  =", round(float(scalar_proj), 4))
print("vector proj  =", np.round(vector_proj, 4))

# --- matrix . vector (Part B) ----------------------------------------------
# Rows of A; x reuses the vector a from Part A.
A = np.array([[${num(A[0])}, ${num(A[1])}],
              [${num(A[2])}, ${num(A[3])}]])
x = a

y = A @ x                         # y[i] = row_i(A) . x   (a dense layer)
print("\\nA =\\n", A, "\\nx =", x)
print("y = A @ x   =", y)
print("  y1 = A[0] . x =", A[0,0]*x[0], "+", A[0,1]*x[1], "=", float(y[0]))
print("  y2 = A[1] . x =", A[1,0]*x[0], "+", A[1,1]*x[1], "=", float(y[1]))

# Columns of A are where the basis vectors land -> y = x1*col1 + x2*col2
print("col1 (A e1) =", A[:, 0], "  col2 (A e2) =", A[:, 1])
print("x1*col1 + x2*col2 =", x[0]*A[:, 0] + x[1]*A[:, 1], " (== y)")
print("det(A)      =", round(float(np.linalg.det(A)), 4), " (|det| = area scale; <0 flips)")
print("shapes: A", A.shape, "@ x", x.shape, "-> y", y.shape)

# --- matrix . matrix (Part C) ----------------------------------------------
B = np.array([[${num(B[0])}, ${num(B[1])}],
              [${num(B[2])}, ${num(B[3])}]])
BA = B @ A                        # apply A first, then B: (BA)_ij = sum_k B_ik A_kj
AB = A @ B                        # apply B first, then A
print("\\nB =\\n", B)
print("B(Ax) =", B @ (A @ x), "   (BA)x =", BA @ x, "  <- the same point (associativity)")
print("A(Bx) =", A @ (B @ x), "   (AB)x =", AB @ x)
print("BA =\\n", BA, "\\nAB =\\n", AB)
comm = float(np.abs(AB - BA).max())
print("max |AB - BA| =", round(comm, 6), "->", "they commute" if comm < 1e-9 else "order matters: AB != BA")
print("unit square under BA has corners", np.round((BA @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]])).T, 4).tolist())
print("det(BA) =", round(float(np.linalg.det(BA)), 4), "= det(B) det(A) =", round(float(np.linalg.det(B) * np.linalg.det(A)), 4))
`;

// ---------------------------------------------------------------------------
// Convex vs non-convex
// ---------------------------------------------------------------------------
export const convexPython = (
  lr: number,
  starts: number[],
  convex: boolean,
) => {
  const def = SURFACES[convex ? 'convex' : 'nonconvex'];
  const fdef = convex
    ? { f: 'x * x', df: '2.0 * x', d2f: '2.0' }
    : { f: '0.15 * x * x + 2.0 * np.sin(3.0 * x)', df: '0.3 * x + 6.0 * np.cos(3.0 * x)', d2f: '0.3 - 18.0 * np.sin(3.0 * x)' };
  const [lo, hi] = def.domain;
  return `import numpy as np

# Convex vs non-convex optimisation -- mirrors the lab.
# Every runner does gradient descent x <- clip(x - alpha f'(x)) in lock-step and
# settles once |f'(x)| < TOL; the lab counts the DISTINCT minima of the SETTLED
# runners, and shades each start's basin of attraction by running the same descent
# from a grid of starts.
# surface: ${ascii(def.label)}
ALPHA = ${num(lr)}
STARTS = [${starts.map((s) => num(s)).join(', ')}]   # the lab's (jittered) starts
DOMAIN = (${num(lo)}, ${num(hi)})
TOL = ${CONVERGE_TOL}
MAX_STEPS = 20000        # the lab keeps stepping until every runner settles; this caps the script
BASIN_MAX_STEPS = ${BASIN_MAX_STEPS}   # the lab's cap for the basin-of-attraction map
CURVE_N = ${CURVE_N}           # basin map / curve samples

def f(x):   return ${fdef.f}
def df(x):  return ${fdef.df}      # analytic gradient
def d2f(x): return ${fdef.d2f}      # curvature: a settled point is a minimum only if d2f > 0

# Local minima of f on the domain (label which basin a runner settled in).
KNOWN_MIN = [${def.minima.map((m) => num(m)).join(', ')}]


def runner_step(x, alpha=ALPHA):
    g = df(x)
    if abs(g) < TOL:
        return x, True
    lo, hi = DOMAIN
    nx = max(lo, min(hi, x - alpha * g))
    return nx, bool(abs(df(nx)) < TOL)


def descend(x0, max_steps):
    x = float(x0)
    for k in range(max_steps):
        if abs(df(x)) < TOL:
            return x, True, k
        x, _ = runner_step(x)
    return x, abs(df(x)) < TOL, max_steps


def basin(x):
    # index of the nearest known minimum -> which basin a settled x belongs to;
    # -1 if it stopped where f'' <= 0 (a maximum, not a minimum)
    if d2f(x) <= 0:
        return -1
    return int(np.argmin([abs(m - x) for m in KNOWN_MIN]))


if __name__ == "__main__":
    xs = [float(s) for s in STARTS]
    settled = [False] * len(xs)
    history = []                          # distinct minima among runners SETTLED at a minimum, per step
    steps = 0
    while not all(settled) and steps < MAX_STEPS:
        for i in range(len(xs)):
            if not settled[i]:
                xs[i], settled[i] = runner_step(xs[i])
        steps += 1
        history.append(len({basin(x) for x, s in zip(xs, settled) if s} - {-1}))
    done = [x for x, s in zip(xs, settled) if s]
    basins = sorted({basin(x) for x in done} - {-1})
    print("starts          :", np.round(STARTS, 3))
    print("settled at      :", np.round(xs, 3), " (settled flags:", settled, ")")
    print("steps           :", steps, "" if all(settled) else "(not all settled -- alpha at or above 2/f'' at the minima?)")
    print("distinct minima :", len(basins), "->", [round(KNOWN_MIN[i], 3) for i in basins])
    print("distinct-minima history (first 20 steps):", history[:20])
    if done:
        best = min(done, key=f)
        print(f"best f found    : f({best:.3f}) = {f(best):.4f}")

    # basin-of-attraction map (the shading under the lab's curve)
    lo, hi = DOMAIN
    grid = [lo + (i / (CURVE_N - 1)) * (hi - lo) for i in range(CURVE_N)]
    labels = []
    for x0 in grid:
        xe, ok, _ = descend(x0, BASIN_MAX_STEPS)
        labels.append(basin(xe) if ok else -1)
    runs, s = [], 0
    for i in range(1, CURVE_N + 1):
        if i == CURVE_N or labels[i] != labels[s]:
            tag = "no minimum reached" if labels[s] < 0 else f"-> min {KNOWN_MIN[labels[s]]}"
            runs.append(f"[{grid[s]:.2f}, {grid[i - 1]:.2f}] {tag}")
            s = i
    print("basins of attraction:", "; ".join(runs))
`;
};

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs — every export × representative
// parameters: lab defaults, every preset / mode the UI can export, slider edges.
// ---------------------------------------------------------------------------
const NC = SURFACES.nonconvex;
const CV = SURFACES.convex;

export const PYTHON_SAMPLES: PythonSample[] = [
  // gradient descent: default, every preset, slider edges
  { name: 'gd-default', code: () => gradientDescentPython('doublewell', 0.05, 'momentum', 0.7, 1.3) },
  { name: 'gd-stable-bowl', code: () => gradientDescentPython('quadratic', 0.1, 'momentum', 0, 2.4) },
  { name: 'gd-divergence', code: () => gradientDescentPython('quadratic', 1.1, 'momentum', 0, 0.5) },
  { name: 'gd-escape-hump', code: () => gradientDescentPython('doublewell', 0.04, 'momentum', 0.85, -1.5) },
  { name: 'gd-newton-1step', code: () => gradientDescentPython('quadratic', 0.1, 'newton', 0, 2.6) },
  { name: 'gd-newton-uphill', code: () => gradientDescentPython('doublewell', 0.1, 'newton', 0, 0.3) },
  { name: 'gd-adam-ripples', code: () => gradientDescentPython('wavy', 0.25, 'adam', 0, 3.6) },
  { name: 'gd-rmsprop-plateau', code: () => gradientDescentPython('doublewell', 0.04, 'rmsprop', 0, 0.05) },
  { name: 'gd-edge-alpha-max-beta-max', code: () => gradientDescentPython('wavy', 1.2, 'momentum', 0.95, -4.2) },
  { name: 'gd-edge-alpha-min-adam', code: () => gradientDescentPython('quadratic', 0.005, 'adam', 0.7, 3) },

  // Taylor / Padé: every function × both modes, every preset, edges
  ...(['sin', 'cos', 'exp', 'geom', 'log', 'tanh', 'runge'] as TaylorFn[]).flatMap((fn) => {
    const [lo, hi] = TAYLOR_FNS[fn].domain;
    const x = Math.max(lo + 0.2, Math.min(hi - 0.2, fn === 'geom' ? 0.5 : fn === 'runge' ? 0.3 : 1.2));
    return (['taylor', 'pade'] as const).map((mode) => ({
      name: `taylor-${fn}-${mode}`,
      code: () => taylorPython(fn, mode, 0, x, 8),
    }));
  }),
  { name: 'taylor-preset-sin-recentre', code: () => taylorPython('sin', 'taylor', 3, 5, 8) },
  { name: 'taylor-preset-geometric-pole', code: () => taylorPython('geom', 'taylor', 0, 0.9, 8) },
  { name: 'taylor-preset-pade-beats-pole', code: () => taylorPython('geom', 'pade', 0, 0.9, 8) },
  { name: 'taylor-preset-runge-edges', code: () => taylorPython('runge', 'taylor', 0, 0.8, 8) },
  { name: 'taylor-preset-tanh-saturation', code: () => taylorPython('tanh', 'pade', 0, 2.5, 8) },
  { name: 'taylor-edge-maxdeg1-pade', code: () => taylorPython('sin', 'pade', 0, 2, 1) },
  { name: 'taylor-edge-maxdeg10-log-boundary', code: () => taylorPython('log', 'taylor', 0, 1, 10) },
  { name: 'taylor-edge-geom-centre-max', code: () => taylorPython('geom', 'pade', 0.7, -1, 10) },
  { name: 'taylor-edge-runge-offcentre', code: () => taylorPython('runge', 'pade', 0.3, 0.8, 10) },

  // linear transformations: every preset × both modes, plus the zero matrix
  ...([
    ['identity', [1, 0, 0, 1]], ['rotation', [0.5, -0.866, 0.866, 0.5]], ['shear', [1, 1, 0, 1]],
    ['scale', [1.6, 0, 0, 0.6]], ['reflection', [1, 0, 0, -1]], ['squash', [1, 0.5, 2, 1]],
    ['rotostretch', [1.2, -0.9, 0.9, 1.2]], ['zero', [0, 0, 0, 0]], ['slider-extreme', [-2.5, 2.5, 2.5, -2.5]],
  ] as [string, [number, number, number, number]][]).flatMap(([nm, m]) => (['eigen', 'svd'] as const).map((mode) => ({
    name: `lintrans-${nm}-${mode}`,
    code: () => linearTransformPython(mode, m[0], m[1], m[2], m[3]),
  }))),

  // eigen & SVD: presets, the reset (identity), singular and zero matrices, slider corners
  ...([
    ['shear', [1, 1, 0, 1]], ['scale', [2, 0, 0, 0.5]], ['rotation', [0, -1, 1, 0]], ['symmetric', [2, 1, 1, 2]],
    ['reflection', [1, 0, 0, -1]], ['identity', [1, 0, 0, 1]], ['rank1', [1, 2, 2, 4]], ['zero', [0, 0, 0, 0]],
    ['corner', [3, 3, 3, 3]], ['neg-scalar', [-1.5, 0, 0, -1.5]],
  ] as [string, [number, number, number, number]][]).map(([nm, m]) => ({
    name: `eigsvd-${nm}`,
    code: () => eigenSvdPython(m[0], m[1], m[2], m[3]),
  })),

  // derivatives: every function × both methods at the defaults, and dx extremes
  ...DERIV_FNS.flatMap((fnd) => (['forward', 'central'] as const).map((method) => ({
    name: `deriv-${fnd.id}-${method}`,
    code: () => derivativesPython(fnd.id, fnd.defaultX0, method, 1),
  }))),
  { name: 'deriv-edge-dx-min', code: () => derivativesPython('sin', 0.6, 'forward', 1e-12) },
  { name: 'deriv-edge-dx-max-central', code: () => derivativesPython('exp', 2, 'central', 2) },
  { name: 'deriv-edge-x0-zero-cubic', code: () => derivativesPython('cubic', 0, 'forward', 1e-8) },

  // chain rule: every preset at its default x0 and at the slider ends
  ...CHAIN_PRESETS.flatMap((p) => [
    { name: `chain-${p.id}`, code: () => chainRulePython(p.id, p.defaultX0) },
    { name: `chain-${p.id}-xmin`, code: () => chainRulePython(p.id, p.xMin) },
    { name: `chain-${p.id}-xmax`, code: () => chainRulePython(p.id, p.xMax) },
  ]),

  // matrix multiplication: defaults, commuting pair, shear B, singular A, zero vectors
  { name: 'matmul-default', code: () => matmulPython([2, 1], [1, 2], [1, 0.5, -0.5, 1], [1.5, 0, 0, 0.5]) },
  { name: 'matmul-commuting', code: () => matmulPython([2, 1], [1, 2], [1, 0.5, -0.5, 1], [0.8, -0.6, 0.6, 0.8]) },
  { name: 'matmul-shear-b', code: () => matmulPython([2, 1], [1, 2], [1, 0.5, -0.5, 1], [1, 1, 0, 1]) },
  { name: 'matmul-singular-a', code: () => matmulPython([1, -1], [0.5, 0.5], [1, 2, 0.5, 1], [1.5, 0, 0, 0.5]) },
  { name: 'matmul-zero-vectors', code: () => matmulPython([0, 0], [0, 0], [-3, 3, 3, -3], [0, 0, 0, 0]) },

  // convex vs non-convex: the lab's first scatter (seed 1), convex surface, extremes
  { name: 'convex-nonconvex-default', code: () => convexPython(0.05, scatterStartsFor(NC, 8, 1), false) },
  { name: 'convex-convex-default', code: () => convexPython(0.05, scatterStartsFor(CV, 8, 2), true) },
  { name: 'convex-nonconvex-16-runners', code: () => convexPython(0.02, scatterStartsFor(NC, 16, 3), false) },
  { name: 'convex-nonconvex-alpha-max', code: () => convexPython(0.3, scatterStartsFor(NC, 4, 4), false), timeoutSec: 120 },
  { name: 'convex-convex-alpha-min', code: () => convexPython(0.005, scatterStartsFor(CV, 2, 5), true) },
];
