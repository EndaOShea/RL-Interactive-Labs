// Runnable Python exports for the Maths labs (template strings — not LLM
// generated), mirroring the on-screen functions and parameters.
import type { Vec2, Mat2 } from './matrix-multiplication';

export type GdOpt = 'momentum' | 'adam' | 'rmsprop' | 'newton';

export const gradientDescentPython = (
  fn: 'quadratic' | 'doublewell' | 'wavy',
  lr: number,
  opt: GdOpt = 'momentum',
) => {
  const defs = {
    quadratic: { f: 'x**2', df: '2*x', d2f: '2.0', dom: '(-3, 3)' },
    doublewell: { f: 'x**4 - x**2', df: '4*x**3 - 2*x', d2f: '12*x**2 - 2', dom: '(-1.7, 1.7)' },
    wavy: { f: '0.15*x**2 + np.sin(3*x)', df: '0.3*x + 3*np.cos(3*x)', d2f: '0.3 - 9*np.sin(3*x)', dom: '(-4.2, 4.2)' },
  }[fn];

  const update = {
    momentum: `        # Heavy-ball momentum: v <- beta*v - alpha*g ; x <- x + v
        v = BETA * v - ALPHA * g
        x = x + v`,
    rmsprop: `        # RMSProp: per-param adaptive step from a decayed mean square gradient
        s = RHO * s + (1 - RHO) * g * g
        x = x - ALPHA * g / (np.sqrt(s) + EPS)
        v = -ALPHA * g / (np.sqrt(s) + EPS)`,
    adam: `        # Adam: bias-corrected first (m) and second (s) moments
        m = B1 * m + (1 - B1) * g
        s = B2 * s + (1 - B2) * g * g
        mhat = m / (1 - B1 ** (t + 1))
        shat = s / (1 - B2 ** (t + 1))
        step = ALPHA * mhat / (np.sqrt(shat) + EPS)
        x = x - step
        v = -step`,
    newton: `        # Newton's method: step by the curvature, x <- x - f'/f''
        h = d2f(x)
        step = g / h if abs(h) > 1e-8 else ALPHA * g
        x = x - step
        v = -step`,
  }[opt];

  return `import numpy as np

# Gradient descent — mirrors the lab (function = ${fn}, optimiser = ${opt})
ALPHA = ${lr}        # learning rate / step scale
BETA  = 0.7          # momentum coefficient
RHO   = 0.9          # RMSProp decay
B1, B2 = 0.9, 0.999  # Adam moment decays
EPS   = 1e-8
STEPS = 200

def f(x):   return ${defs.f}
def df(x):  return ${defs.df}      # analytic 1st derivative
def d2f(x): return ${defs.d2f}     # analytic 2nd derivative (Hessian in 1-D)

def descend(x0, steps=STEPS, tol=1e-3):
    x, v, m, s = float(x0), 0.0, 0.0, 0.0
    for t in range(steps):
        g = df(x)
${update}
        if abs(df(x)) < tol and abs(v) < tol:
            print(f"converged at step {t}")
            break
    return x

if __name__ == "__main__":
    # domain ${defs.dom}
    x_star = descend(1.3)
    print(f"x*={x_star:.4f}  f(x*)={f(x_star):.4f}  f'(x*)={df(x_star):.2e}")
`;
};

export const taylorPython = (
  fn: 'sin' | 'cos' | 'exp' | 'geom' | 'log' | 'tanh' | 'runge',
  mode: 'taylor' | 'pade' = 'taylor',
) => {
  const f = {
    sin: 'np.sin(x)',
    cos: 'np.cos(x)',
    exp: 'np.exp(x)',
    geom: '1.0 / (1.0 - x)',
    log: 'np.log(1.0 + x)',
    tanh: 'np.tanh(x)',
    runge: '1.0 / (1.0 + 25.0 * x * x)',
  }[fn];

  const padeBlock = mode === 'pade' ? `

def pade(coeffs, m):
    # Build the [m/m] Pade approximant from Taylor coefficients c[0..2m].
    # Solve for denominator coeffs b (b0 = 1), then numerator a.
    import numpy as _np
    A = _np.zeros((m, m)); rhs = _np.zeros(m)
    for i in range(m):
        for j in range(m):
            A[i, j] = coeffs[m + i - j] if 0 <= m + i - j < len(coeffs) else 0.0
        rhs[i] = -coeffs[m + i + 1]
    b = _np.linalg.solve(A, rhs)
    b = _np.concatenate(([1.0], b))
    a = _np.zeros(m + 1)
    for i in range(m + 1):
        a[i] = sum(coeffs[i - k] * b[k] for k in range(min(i, m) + 1))
    return a, b

def pade_eval(a, b, x):
    num = sum(a[i] * x ** i for i in range(len(a)))
    den = sum(b[i] * x ** i for i in range(len(b)))
    return num / den
` : '';

  const main = mode === 'pade'
    ? `if __name__ == "__main__":
    c = [deriv(f, A, k) / factorial(k) for k in range(N + 1)]
    a, b = pade([deriv(f, 0.0, k) / factorial(k) for k in range(2 * (N // 2) + 1)], N // 2)
    approx = pade_eval(a, b, EVAL)
    print(f"f({EVAL})={f(EVAL):.6f}  Pade[{N//2}/{N//2}]={approx:.6f}  err={abs(f(EVAL)-approx):.2e}")`
    : `if __name__ == "__main__":
    approx = taylor(EVAL)
    print(f"f({EVAL})={f(EVAL):.6f}  T_{N}({EVAL})={approx:.6f}  err={abs(f(EVAL)-approx):.2e}")`;

  return `import numpy as np
from math import factorial

# Polynomial / rational approximation — mirrors the lab
# (function = ${fn}, mode = ${mode})
# Taylor:  T_n(x) = sum_{k=0}^{n} f^(k)(a)/k! * (x - a)^k
# Pade:    rational [m/m] approximant fit to the same Taylor coefficients
A = 0.0          # expansion centre
N = 8            # max degree
EVAL = 2.0

def f(x): return ${f}

def deriv(g, x, k, h=1e-3):
    # central finite-difference k-th derivative (analytic in the lab)
    if k == 0:
        return g(x)
    return (deriv(g, x + h, k - 1, h) - deriv(g, x - h, k - 1, h)) / (2 * h)

def taylor(x, a=A, n=N):
    return sum(deriv(f, a, k) / factorial(k) * (x - a) ** k for k in range(n + 1))
${padeBlock}
${main}
`;
};

export const linearTransformPython = (
  mode: 'eigen' | 'svd' = 'eigen',
) => `import numpy as np

# Linear transformations of the plane — mirrors the lab (mode = ${mode})
# Columns of M are where the basis vectors i-hat, j-hat land.
M = np.array([[1.0, 1.0],
              [0.0, 1.0]])   # shear; try rotation / scale / reflection

def analyse(M):
    det = np.linalg.det(M)            # signed area scale
    vals, vecs = np.linalg.eig(M)     # eigenvalues / eigenvectors (may be complex)
    return det, vals, vecs

def svd_analyse(M):
    # M = U @ diag(s) @ Vt : rotate (Vt) -> stretch (s) -> rotate (U).
    U, s, Vt = np.linalg.svd(M)
    cond = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    return U, s, Vt, cond

if __name__ == "__main__":
    det, vals, vecs = analyse(M)
    print("det     =", round(float(det), 4), "(area scale; <0 flips orientation)")
    print("eigvals =", vals)
    print("eigvecs =\\n", vecs)

    U, s, Vt, cond = svd_analyse(M)
    print("singular values =", np.round(s, 4), " (always real, >= 0)")
    print("condition number =", round(float(cond), 4))
    print("U =\\n", np.round(U, 4), "\\nVt =\\n", np.round(Vt, 4))

    # apply to a vector
    v = np.array([1.0, 0.5])
    print("M v     =", M @ v)
`;


export const derivativesPython = (
  fn: string = 'square',
  x0: number = 1.0,
) => {
  const defs: Record<string, { f: string; df: string; name: string }> = {
    square: { f: 'x**2', df: '2*x', name: 'x^2' },
    cubic: { f: 'x**3 - x', df: '3*x**2 - 1', name: 'x^3 - x' },
    sin: { f: 'np.sin(x)', df: 'np.cos(x)', name: 'sin(x)' },
    exp: { f: 'np.exp(x)', df: 'np.exp(x)', name: 'e^x' },
  };
  const d = defs[fn] || defs.square;

  return `import numpy as np

# Derivative as the limit of a secant slope -- mirrors the lab.
# f'(x) = lim_{dx->0} [f(x + dx) - f(x)] / dx
# Function = ${d.name},  point x0 = ${x0}
X0 = ${x0}

def f(x):  return ${d.f}
def df(x): return ${d.df}      # analytic derivative (known closed form)

analytic = df(X0)
print(f"f(x) = ${d.name}   f'(x0) analytic = {analytic:.6f}\\n")
print(f"{'dx':>10}  {'secant slope':>14}  {'|error|':>12}")

prev = None
for dx in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5]:
    secant = (f(X0 + dx) - f(X0)) / dx       # forward difference
    err = abs(secant - analytic)
    ratio = f"  (err x{prev / err:5.1f})" if prev and err > 0 else ""
    print(f"{dx:>10.5f}  {secant:>14.8f}  {err:>12.3e}{ratio}")
    prev = err

# The forward-difference error ~ 0.5 * f''(x0) * dx, so halving dx roughly
# halves the error -- the secant slope converges to the tangent slope f'(x0).
`;
};

export const chainRulePython = (
  preset: 'sin_sq' | 'poly_sq' | 'gauss' | 'logistic' = 'sin_sq',
  x0 = 1.2,
) => {
  // Each preset: the composite, plus the ordered links (forward value + analytic
  // local derivative). The product of the local derivatives is dy/dx, checked
  // against a central finite difference of the whole composite.
  const presets: Record<string, { formula: string; composite: string; links: string }> = {
    sin_sq: {
      formula: 'y = sin(x^2)',
      composite: 'np.sin(x * x)',
      links: `    [
        ("u", "x",  lambda x: x * x,      lambda x: 2 * x,        "u = x^2",   "du/dx = 2x"),
        ("y", "u",  lambda u: np.sin(u),  lambda u: np.cos(u),    "y = sin(u)","dy/du = cos(u)"),
    ]`,
    },
    poly_sq: {
      formula: 'y = (3x + 1)^2',
      composite: '(3 * x + 1) ** 2',
      links: `    [
        ("u", "x",  lambda x: 3 * x + 1,  lambda x: 3.0,          "u = 3x + 1","du/dx = 3"),
        ("y", "u",  lambda u: u * u,      lambda u: 2 * u,        "y = u^2",   "dy/du = 2u"),
    ]`,
    },
    gauss: {
      formula: 'y = exp(-x^2)',
      composite: 'np.exp(-(x * x))',
      links: `    [
        ("u", "x",  lambda x: x * x,      lambda x: 2 * x,        "u = x^2",   "du/dx = 2x"),
        ("v", "u",  lambda u: -u,         lambda u: -1.0,         "v = -u",    "dv/du = -1"),
        ("y", "v",  lambda v: np.exp(v),  lambda v: np.exp(v),    "y = exp(v)","dy/dv = exp(v)"),
    ]`,
    },
    logistic: {
      formula: 'y = 1 / (1 + e^(-2x))',
      composite: '1.0 / (1.0 + np.exp(-2 * x))',
      links: `    [
        ("u", "x",  lambda x: -2 * x,           lambda x: -2.0,         "u = -2x",   "du/dx = -2"),
        ("v", "u",  lambda u: 1 + np.exp(u),     lambda u: np.exp(u),    "v = 1 + e^u","dv/du = e^u"),
        ("y", "v",  lambda v: 1.0 / v,           lambda v: -1.0 / (v*v), "y = 1 / v", "dy/dv = -1/v^2"),
    ]`,
    },
  };
  const p = presets[preset] || presets.sin_sq;

  return `import numpy as np

# Chain rule — mirrors the lab (composite = ${p.formula})
# dy/dx is the PRODUCT of each link's local derivative along the path x -> ... -> y.
X0 = ${x0}

def composite(x):
    return ${p.composite}

# Ordered links: (out_name, in_name, value_fn, local_deriv_fn, expr, deriv_expr)
LINKS = ${p.links}

def chain_rule(x0):
    val = x0
    product = 1.0
    factors = []
    print(f"forward:  x = {x0:.4f}")
    for out, inp, value_fn, dfn, expr, dexpr = (None,) * 6 if False else (None,):  # placeholder
        pass
    return product, factors

def evaluate(x0):
    val = x0
    product = 1.0
    factors = []
    print(f"forward:  x = {x0:.4f}")
    for out, inp, value_fn, dfn, expr, dexpr in LINKS:
        local = dfn(val)               # local derivative at THIS link's input
        product *= local
        factors.append((dexpr, local))
        val = value_fn(val)            # forward value carried to the next link
        print(f"forward:  {expr:<12} -> {out} = {val:.4f}   ({dexpr} = {local:.4f})")
    return product, factors

def finite_diff(x0, h=1e-5):
    # Independent cross-check: central difference of the WHOLE composite.
    return (composite(x0 + h) - composite(x0 - h)) / (2 * h)

if __name__ == "__main__":
    product, factors = evaluate(X0)
    chain = "  x  ".join(f"{v:.4f}" for _, v in reversed(factors)).replace("  x  ", " * ")
    print()
    print("dy/dx = " + " * ".join(f"{v:.4f}" for _, v in reversed(factors)) + f" = {product:.6f}")
    fd = finite_diff(X0)
    print(f"finite-difference check  = {fd:.6f}")
    print(f"agree: {abs(product - fd) < 1e-3}")
`;
};

export const matmulPython = (
  a: Vec2,
  b: Vec2,
  A: Mat2,
) => `import numpy as np

# Matrix multiplication — mirrors the lab.
# Part A: the dot product (a.b)        — the operation every neuron computes.
# Part B: matrix . vector (y = A @ x)  — the operation every dense layer performs.

# --- vectors (Part A) -------------------------------------------------------
a = np.array([${a[0]}, ${a[1]}])
b = np.array([${b[0]}, ${b[1]}])

dot = a @ b                       # = a[0]*b[0] + a[1]*b[1]
na, nb = np.linalg.norm(a), np.linalg.norm(b)
cos_t = dot / (na * nb)           # a.b = |a||b| cos(theta)
theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))

# projection of a onto b
scalar_proj = dot / nb            # signed length of a's shadow on b
vector_proj = (dot / (b @ b)) * b

print("a . b        =", dot, "= a1*b1 + a2*b2 =", a[0]*b[0], "+", a[1]*b[1])
print("|a||b|cos(t) =", round(float(na * nb * cos_t), 4), " (same number, geometric form)")
print("cos(theta)   =", round(float(cos_t), 4), " theta =", round(float(theta), 1), "deg")
print("scalar proj  =", round(float(scalar_proj), 4))
print("vector proj  =", np.round(vector_proj, 4))

# --- matrix . vector (Part B) ----------------------------------------------
# Rows of A; x reuses the vector a from Part A.
A = np.array([[${A[0]}, ${A[1]}],
              [${A[2]}, ${A[3]}]])
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

# Shapes / compatibility: (m x n) @ (n,) -> (m,). Inner dims must match.
print("shapes: A", A.shape, "@ x", x.shape, "-> y", y.shape)
`;

export const convexPython = (
  lr: number,
  nStarts: number,
  convex: boolean,
) => {
  const fdef = convex
    ? { f: 'x * x', df: '2.0 * x', known: '[0.0]', label: 'convex  f(x) = x^2' }
    : {
        f: '0.15 * x * x + 2.0 * np.sin(3.0 * x)',
        df: '0.3 * x + 6.0 * np.cos(3.0 * x)',
        known: '[-2.5751, -0.5152, 1.5447, 3.6048]',
        label: 'non-convex  f(x) = 0.15 x^2 + 2 sin 3x',
      };

  return `import numpy as np

# Convex vs Non-convex optimisation - mirrors the lab.
# Multi-start gradient descent on f over a fixed domain, then count the
# DISTINCT minima the runners settle in (snapped to the known minima of f).
# surface = ${convex ? 'convex' : 'non-convex'}
ALPHA    = ${lr}        # learning rate / step size
N_STARTS = ${nStarts}        # independent gradient-descent runners
DOMAIN   = (-4.0, 4.0)
TOL      = 1e-3        # |f'(x)| below this -> settled
MAX_STEPS = 600

def f(x):  return ${fdef.f}
def df(x): return ${fdef.df}      # analytic gradient

# Local minima of f on the domain (used to label which basin a runner lands in).
KNOWN_MIN = ${fdef.known}

def descend(x0, alpha=ALPHA, steps=MAX_STEPS, tol=TOL):
    # x <- x - alpha * f'(x), clipped to the domain, until the slope vanishes.
    x = float(x0)
    lo, hi = DOMAIN
    for _ in range(steps):
        g = df(x)
        if abs(g) < tol:
            break
        x = min(hi, max(lo, x - alpha * g))
    return x

def basin(x):
    # index of the nearest known minimum -> which basin x settled in
    return int(np.argmin([abs(m - x) for m in KNOWN_MIN]))

if __name__ == "__main__":
    lo, hi = DOMAIN
    starts = np.linspace(lo + 0.35, hi - 0.35, N_STARTS)
    finals = np.array([descend(s) for s in starts])

    basins = sorted(set(basin(x) for x in finals))
    best = finals[int(np.argmin([f(x) for x in finals]))]

    print("surface         :", "${fdef.label}")
    print("starts          :", np.round(starts, 3))
    print("settled at      :", np.round(finals, 3))
    print("distinct minima :", len(basins),
          "->", [round(KNOWN_MIN[i], 3) for i in basins])
    print(f"best f found    : f({best:.3f}) = {f(best):.4f}")
    if len(basins) == 1:
        print("=> all runners agree: initialisation did not matter (convex-like single basin).")
    else:
        print("=> runners split across basins: the result DEPENDS on initialisation (non-convex).")
`;
};

export const eigenSvdPython = (
  a: number,
  b: number,
  c: number,
  d: number,
) => `import numpy as np

# Eigenvalues & SVD of a 2x2 matrix - mirrors the lab.
# A maps the unit circle to an ellipse; SVD reads A = U @ diag(s) @ Vt as
# rotate (Vt) -> scale (s) -> rotate (U).
A = np.array([[${a}, ${b}],
              [${c}, ${d}]])

def eigen(A):
    # Characteristic equation: lambda^2 - t*lambda + det = 0
    t   = np.trace(A)              # a + d
    det = np.linalg.det(A)         # ad - bc
    disc = t * t - 4 * det
    vals, vecs = np.linalg.eig(A)  # may be complex when disc < 0 (a rotation)
    return t, det, disc, vals, vecs

def svd(A):
    # s are sqrt of the eigenvalues of A^T A; always real and >= 0.
    U, s, Vt = np.linalg.svd(A)
    cond = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    return U, s, Vt, cond

if __name__ == "__main__":
    t, det, disc, vals, vecs = eigen(A)
    print(f"trace = {t:.4f}   det = {det:.4f}   disc = t^2 - 4det = {disc:.4f}")
    if disc < 0:
        print("disc < 0 -> complex eigenvalues (a rotation, no real invariant axis)")
    print("eigenvalues  =", np.round(vals, 4))
    print("eigenvectors =\\n", np.round(vecs, 4), " (columns; A v = lambda v)")

    U, s, Vt, cond = svd(A)
    print("\\nA = U @ diag(s) @ Vt")
    print("singular values s =", np.round(s, 4), " (real, >= 0; ellipse semi-axes)")
    print("condition number  = sigma1/sigma2 =",
          round(float(cond), 4) if np.isfinite(cond) else "inf")
    print("U  =\\n", np.round(U, 4))
    print("Vt =\\n", np.round(Vt, 4))

    # sanity: reconstruct A and map the unit circle to its ellipse
    recon = U @ np.diag(s) @ Vt
    print("\\nreconstruction error |A - U S Vt| =", np.round(np.abs(A - recon).max(), 8))
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)])   # 2 x 64 unit circle
    ellipse = A @ circle                                 # its image
    print("ellipse semi-axis lengths (== s):", np.round(np.sort(s)[::-1], 4))
`;