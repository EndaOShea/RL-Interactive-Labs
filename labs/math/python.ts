// Runnable Python exports for the Maths labs (template strings — not LLM
// generated), mirroring the on-screen functions and parameters.

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
