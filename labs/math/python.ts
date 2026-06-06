// Runnable Python exports for the Maths labs (template strings — not LLM
// generated), mirroring the on-screen functions and parameters.

export const gradientDescentPython = (
  fn: 'quadratic' | 'doublewell' | 'wavy',
  lr: number,
) => {
  const defs = {
    quadratic: { f: 'x**2', df: '2*x', dom: '(-3, 3)' },
    doublewell: { f: 'x**4 - x**2', df: '4*x**3 - 2*x', dom: '(-1.7, 1.7)' },
    wavy: { f: '0.15*x**2 + np.sin(3*x)', df: '0.3*x + 3*np.cos(3*x)', dom: '(-4.2, 4.2)' },
  }[fn];
  return `import numpy as np

# Gradient descent with momentum — mirrors the lab (function = ${fn})
ALPHA = ${lr}        # learning rate
BETA  = 0.7          # momentum (set 0 for plain GD)
STEPS = 200

def f(x):  return ${defs.f}
def df(x): return ${defs.df}     # analytic derivative

def descend(x0, alpha=ALPHA, beta=BETA, steps=STEPS, tol=1e-3):
    x, v = float(x0), 0.0
    for t in range(steps):
        g = df(x)
        v = beta * v - alpha * g
        x = x + v
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
  fn: 'sin' | 'cos' | 'exp' | 'geom' | 'log',
) => {
  const f = {
    sin: 'np.sin(x)',
    cos: 'np.cos(x)',
    exp: 'np.exp(x)',
    geom: '1.0 / (1.0 - x)',
    log: 'np.log(1.0 + x)',
  }[fn];
  return `import numpy as np
from math import factorial

# Taylor series approximation — mirrors the lab (function = ${fn})
# T_n(x) = sum_{k=0}^{n} f^(k)(a)/k! * (x - a)^k
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

if __name__ == "__main__":
    approx = taylor(EVAL)
    print(f"f({EVAL})={f(EVAL):.6f}  T_{N}({EVAL})={approx:.6f}  err={abs(f(EVAL)-approx):.2e}")
`;
};

export const linearTransformPython = () => `import numpy as np

# Linear transformations of the plane — mirrors the lab
# Columns of M are where the basis vectors i-hat, j-hat land.
M = np.array([[1.0, 1.0],
              [0.0, 1.0]])   # shear; try rotation / scale / reflection

def analyse(M):
    det = np.linalg.det(M)            # signed area scale
    vals, vecs = np.linalg.eig(M)     # eigenvalues / eigenvectors (may be complex)
    return det, vals, vecs

if __name__ == "__main__":
    det, vals, vecs = analyse(M)
    print("det     =", round(float(det), 4), "(area scale; <0 flips orientation)")
    print("eigvals =", vals)
    print("eigvecs =\\n", vecs)
    # apply to a vector
    v = np.array([1.0, 0.5])
    print("M v     =", M @ v)
`;
