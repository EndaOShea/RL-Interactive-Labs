// Runnable Python exports for the Stochastic & Bayesian Models labs (template
// strings — not LLM-generated). Each mirrors its lab exactly: the seeded
// generator (mulberry32 + Box–Muller, ported bit-for-bit) reproduces the lab's
// random feature layers, training data and dice sequence; the parameters are the
// live slider values at download time; and the algorithms are the lab's.
import type { PythonSample } from '../../utils/pythonSamples';
import type { BnnMode } from './bnnCore';
import type { KernelId } from './shared';

export type { BnnMode } from './bnnCore';
export type { KernelId } from './shared';

const py = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');

/** The lab's seeded PRNG (mulberry32) and Box–Muller normal, as Python. */
const PY_RNG = `def mulberry32(seed):
    # the lab's seeded PRNG, in unsigned 32-bit arithmetic (bit-identical output)
    state = [seed & 0xFFFFFFFF]
    def r():
        a = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        state[0] = a
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

def gauss_from(r):
    # Box-Muller from a uniform source, as in the lab
    u = 0.0
    while u == 0.0:
        u = r()
    v = 0.0
    while v == 0.0:
        v = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)`;

// ---------------------------------------------------------------------------
// Bayesian neural network
// ---------------------------------------------------------------------------

export const bnnPython = (mode: BnnMode, alpha: number, noise: number) => `import math
import numpy as np

# Bayesian neural network = a fixed random tanh feature layer + a Bayesian linear
# output layer (mirrors the lab; mode = ${mode}). The exact posterior over the
# output weights is Gaussian, so every approximation can be compared with it.
MODE = "${mode}"
M = 24                         # random tanh features
ALPHA = ${py(alpha)}                  # prior precision over the output weights
NOISE = ${py(noise)}                  # observation noise std
BETA = 1 / NOISE ** 2          # noise precision
P_DROP = 0.2                   # MC-dropout rate
ENS_K = 8                      # deep-ensemble members (feature seeds 100..107)
N_DRAWS = {"point": 1, "ensemble": ENS_K}.get(MODE, 28)
rng = np.random.default_rng(0)  # Monte Carlo draws (the lab uses an unseeded source)

${PY_RNG}

def make_features(m, seed):
    # phi(x) = tanh(w1 x + b1): w1 ~ 4.5 N(0,1), b1 ~ U(-3, 3), from the lab's seeded generator
    r = mulberry32(seed)
    w1 = np.array([gauss_from(r) * 4.5 for _ in range(m)])
    b1 = np.array([(r() * 2 - 1) * 3 for _ in range(m)])
    return w1, b1

def design(xs, w1, b1):
    return np.tanh(np.outer(xs, w1) + b1)

def f_true(x):
    return 0.8 * math.sin(2 * math.pi * 1.3 * x)

# Training data with a GAP in the middle (the lab's seed 20250608, noise 0.05).
r = mulberry32(20250608)
Xtr = [0.05 + (i / 6) * 0.30 for i in range(7)] + [0.62 + (i / 6) * 0.33 for i in range(7)]
ytr = np.array([f_true(x) + gauss_from(r) * 0.05 for x in Xtr])
Xtr = np.array(Xtr)

xs = np.arange(161) / 160            # plotting grid: x = 0.2 and 0.5 are grid points
IDX_DATA, IDX_GAP = 32, 80

def posterior(Phi):
    A = ALPHA * np.eye(M) + BETA * Phi.T @ Phi          # posterior precision
    S = np.linalg.inv(A)                                  # Sigma
    m = BETA * S @ Phi.T @ ytr                            # mean
    return A, S, m

w1, b1 = make_features(M, 7)
Phi, Phis = design(Xtr, w1, b1), design(xs, w1, b1)
A, S, m = posterior(Phi)
exact_sd = np.sqrt(np.einsum('ij,jk,ik->i', Phis, S, Phis))  # exact epistemic sd of f(x)

# ---- mean-field VI: the ELBO optimum is q_i = N(m_i, 1/A_ii) ----
s2 = 1 / np.diag(A)
def elbo(mu, s2):
    n = len(ytr)
    rss = np.sum((ytr - Phi @ mu) ** 2)
    trace = np.sum(s2 * np.sum(Phi ** 2, axis=0))
    lik = n / 2 * math.log(BETA / (2 * math.pi)) - BETA / 2 * (rss + trace)
    prior = M / 2 * math.log(ALPHA / (2 * math.pi)) - ALPHA / 2 * (mu @ mu + s2.sum())
    return lik + prior + np.sum(0.5 * np.log(2 * math.pi * math.e * s2))
def log_evidence():
    n = len(ytr)
    E = BETA / 2 * np.sum((ytr - Phi @ m) ** 2) + ALPHA / 2 * m @ m
    return M / 2 * math.log(ALPHA) + n / 2 * math.log(BETA) - E - 0.5 * np.linalg.slogdet(A)[1] - n / 2 * math.log(2 * math.pi)

# ---- MC-dropout: fit the output layer under the dropout-regularised objective ----
PtP = Phi.T @ Phi
k = P_DROP / (1 - P_DROP)
w_drop = np.linalg.solve(BETA * PtP + BETA * k * np.diag(np.diag(PtP)) + ALPHA * np.eye(M), BETA * Phi.T @ ytr)

def draw(i):
    if MODE == "point":
        return Phis @ m
    if MODE == "exact":
        L = np.linalg.cholesky(S + 1e-9 * np.eye(M))
        return Phis @ (m + L @ rng.standard_normal(M))
    if MODE == "variational":
        return Phis @ (m + np.sqrt(s2) * rng.standard_normal(M))
    if MODE == "dropout":
        mask = rng.random(M) > P_DROP                      # keep each feature with prob 1 - p
        return Phis @ (mask * w_drop / (1 - P_DROP))
    # deep ensemble: member i has its own random feature layer, fitted to the same data
    w1k, b1k = make_features(M, 100 + i % ENS_K)
    _, _, mk = posterior(design(Xtr, w1k, b1k))
    return design(xs, w1k, b1k) @ mk

if __name__ == "__main__":
    curves = np.stack([draw(i) for i in range(N_DRAWS)])
    sd = curves.std(axis=0)                                  # epistemic spread of the drawn curves
    for name, g in (("gap  x=0.5", IDX_GAP), ("data x=0.2", IDX_DATA)):
        pred = math.sqrt(sd[g] ** 2 + 1 / BETA)              # predictive: + noise variance (tau^-1)
        print(f"{name}: sigma_f from {N_DRAWS} curves = {sd[g]:.4f}   exact = {exact_sd[g]:.4f}   predictive sigma_y = {pred:.4f}")
    if MODE == "variational":
        lz, e = log_evidence(), elbo(m, s2)
        print(f"ELBO = {e:.3f}   log evidence = {lz:.3f}   KL(q||p) = {lz - e:.3f} nats")
        ratio = np.sort(s2 / np.diag(S))[M // 2]             # the lab's median (upper of the two middle values)
        print(f"median weight-variance ratio s_i^2 / Sigma_ii = {ratio:.2e}  (mean-field shrinks it {1 / ratio:.0f}x)")
    if MODE == "dropout":
        print(f"|w_drop| = {np.linalg.norm(w_drop):.3f}  vs  |m| = {np.linalg.norm(m):.3f}  (the dropout penalty shrinks the fit)")
`;

// ---------------------------------------------------------------------------
// Gaussian process
// ---------------------------------------------------------------------------

export const gpPython = (k: KernelId, ell: number, sf: number, sn: number, revealed = 8) => `import math
import numpy as np

# Gaussian-process regression (mirrors the lab; kernel = ${k}).
KERNEL = "${k}"
ELL, SF, SN = ${py(ell)}, ${py(sf)}, ${py(sn)}       # lengthscale, signal std, noise std
PERIOD = ${py(0.3)}                     # period of the periodic kernel
REVEALED = ${Math.max(0, Math.min(8, Math.round(revealed)))}                    # observations conditioned on (the lab reveals them one by one)
SAMPLE_JITTER = 1e-6

${PY_RNG}

def f_true(x):
    return math.sin(2 * math.pi * x) * 0.7

# The lab's 8 observations with a gap: y = f(x) + 0.06 N(0,1), generator seed 7.
r = mulberry32(7)
X_ALL = np.array([0.07, 0.16, 0.25, 0.33, 0.60, 0.70, 0.82, 0.93])
Y_ALL = np.array([f_true(x) + gauss_from(r) * 0.06 for x in X_ALL])
X, y = X_ALL[:REVEALED], Y_ALL[:REVEALED]

def kern(a, b):
    r = np.abs(a[:, None] - b[None, :])
    v = SF ** 2
    if KERNEL == "rbf":
        return v * np.exp(-(r * r) / (2 * ELL * ELL))
    if KERNEL == "matern32":
        s = math.sqrt(3) * r / ELL
        return v * (1 + s) * np.exp(-s)
    if KERNEL == "periodic":
        s = np.sin(math.pi * r / PERIOD)
        return v * np.exp((-2 * s * s) / (ELL * ELL))
    # linear: straight lines through x = 0.5, plus a 0.02 sf^2 bias (offset) term; ELL is unused
    return v * ((a[:, None] - 0.5) * (b[None, :] - 0.5)) + 0.02 * v

xs = np.arange(140) / 139            # fine grid for the mean and band
xc = np.arange(56) / 55              # coarse grid for sample functions
IDX_GAP, IDX_DATA = 64, 22           # x = 0.460 (in the gap), 0.158 (beside the point at 0.16)

if REVEALED == 0:
    mean = np.zeros(xs.size)
    std = np.sqrt(np.diag(kern(xs, xs)))
    mean_c, cov_c = np.zeros(xc.size), kern(xc, xc)
    log_ml = None
else:
    L = np.linalg.cholesky(kern(X, X) + SN ** 2 * np.eye(X.size))   # K + sn^2 I = L L^T
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))             # two triangular solves, no inverse
    Ks = kern(xs, X)
    mean = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    std = np.sqrt(np.clip(np.diag(kern(xs, xs)) - np.sum(v * v, axis=0), 0, None))
    Kc = kern(xc, X)
    mean_c = Kc @ alpha
    vc = np.linalg.solve(L, Kc.T)
    cov_c = kern(xc, xc) - vc.T @ vc
    # log marginal likelihood: -1/2 y^T (K + sn^2 I)^-1 y - sum log L_ii - n/2 log 2 pi
    log_ml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - X.size / 2 * math.log(2 * math.pi)

# a few sample functions on the coarse grid (prior when nothing is revealed)
Lc = np.linalg.cholesky(cov_c + SAMPLE_JITTER * np.eye(xc.size))
samples = mean_c[:, None] + Lc @ np.random.default_rng(2).standard_normal((xc.size, 4))

def fit_hyperparameters():
    # The lab's "fit" button: coordinate ascent of the log marginal likelihood over
    # the slider grids (ell, then sf, then sn), on all 8 points, from the current values.
    global ELL, SF, SN
    grid = lambda lo, hi, st: [round(lo + i * st, 4) for i in range(int(round((hi - lo) / st)) + 1)]
    def lml(l, f, s):
        global ELL, SF
        ELL, SF = l, f
        Lf = np.linalg.cholesky(kern(X_ALL, X_ALL) + s ** 2 * np.eye(X_ALL.size))
        a = np.linalg.solve(Lf.T, np.linalg.solve(Lf, Y_ALL))
        return -0.5 * Y_ALL @ a - np.sum(np.log(np.diag(Lf))) - X_ALL.size / 2 * math.log(2 * math.pi)
    l, f, s = ELL, SF, SN
    best = lml(l, f, s)
    for _ in range(30):
        before = best
        for lv in grid(0.03, 0.6, 0.005):
            val = lml(lv, f, s)
            if val > best: best, l = val, lv
        for fv in grid(0.3, 2, 0.05):
            val = lml(l, fv, s)
            if val > best: best, f = val, fv
        for sv in grid(0.01, 0.3, 0.005):
            val = lml(l, f, sv)
            if val > best: best, s = val, sv
        if best - before < 1e-9:
            break
    ELL, SF, SN = l, f, s
    return l, f, s, best

if __name__ == "__main__":
    print(f"conditioned on {REVEALED} of 8 points")
    print(f"posterior std beside the data (x={xs[IDX_DATA]:.3f}): {std[IDX_DATA]:.3f}")
    print(f"posterior std in the gap      (x={xs[IDX_GAP]:.3f}): {std[IDX_GAP]:.3f}")
    print("log marginal likelihood:", "undefined (no data)" if log_ml is None else round(float(log_ml), 3))
    l, f, s, best = fit_hyperparameters()
    print(f"max-log-ML hyperparameters on the slider grid: ell={l}, sf={f}, sn={s} -> log ML {best:.3f}")
`;

// ---------------------------------------------------------------------------
// Hidden Markov model
// ---------------------------------------------------------------------------

export const hmmPython = (selfStay: number, loadedSix: number, length: number) => `import math
import numpy as np

# Hidden Markov model — the occasionally-dishonest casino (mirrors the lab).
# Two hidden states: 0 = Fair die, 1 = Loaded die. Observations are die rolls 1..6.
STAY = ${py(selfStay)}                 # P(stay in the same die)
A = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])
pi = np.array([0.5, 0.5])

fair = np.full(6, 1 / 6)
p6 = ${py(loadedSix)}                   # P(roll a 6) on the loaded die
loaded = np.r_[np.full(5, (1 - p6) / 5), p6]
B = np.stack([fair, loaded])                   # 2 x 6 emission matrix

${PY_RNG}

# The lab's sequence: generated from the true model with its seeded generator (seed 12345).
T = ${Math.max(1, Math.round(length))}
r = mulberry32(12345)
def pick(p):
    u, acc = r(), 0.0
    for i, q in enumerate(p):
        acc += q
        if u <= acc:
            return i
    return len(p) - 1
states, obs = [], []
s = pick(pi)
for _ in range(T):
    states.append(s)
    obs.append(pick(B[s]))
    s = pick(A[s])
obs = np.array(obs)

def forward(obs):
    # scaled forward pass: alpha_t is the filtered posterior; sum(log c_t) = log p(obs)
    T = obs.size
    alpha = np.zeros((T, 2)); c = np.zeros(T)
    alpha[0] = pi * B[:, obs[0]]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    return alpha, c

def forward_backward(obs):
    T = obs.size
    alpha, c = forward(obs)
    beta = np.zeros((T, 2)); beta[-1] = 1
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
    g = alpha * beta
    return g / g.sum(1, keepdims=True)                 # smoothed posterior P(state|all obs)

def viterbi(obs):
    T = obs.size
    d = np.zeros((T, 2)); psi = np.zeros((T, 2), int)
    d[0] = np.log(pi) + np.log(B[:, obs[0]])
    for t in range(1, T):
        for j in range(2):
            seq = d[t - 1] + np.log(A[:, j])
            psi[t, j] = seq.argmax(); d[t, j] = seq.max() + np.log(B[j, obs[t]])
    path = [int(d[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    return path[::-1]

if __name__ == "__main__":
    alpha, c = forward(obs)
    post = forward_backward(obs)
    path = viterbi(obs)
    print("rolls           :", "".join(str(o + 1) for o in obs))
    print("true die        :", "".join(str(s) for s in states))
    print("Viterbi path    :", "".join(str(s) for s in path))
    print("Viterbi state-recovery accuracy:", round(float(np.mean(np.array(path) == np.array(states))), 3))
    print(f"log p(rolls) = {np.log(c).sum():.3f}   (all-fair model: {T * math.log(1 / 6):.3f})")
    print("filtered P(loaded) :", np.round(alpha[:, 1], 2))
    print("smoothed P(loaded) :", np.round(post[:, 1], 2))
    decode = (post[:, 1] > 0.5).astype(int)
    print("per-roll most likely die (smoothed) disagrees with Viterbi at",
          int(np.sum(decode != np.array(path))), "of", T, "rolls")
`;

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs
// ---------------------------------------------------------------------------

const BNN_MODES: BnnMode[] = ['exact', 'variational', 'dropout', 'ensemble', 'point'];
const KERNEL_IDS: KernelId[] = ['rbf', 'matern32', 'periodic', 'linear'];

export const PYTHON_SAMPLES: PythonSample[] = [
  // Bayesian NN — every mode at the defaults, and the slider corners
  ...BNN_MODES.map((m) => ({ name: `bnn-${m}-default`, code: () => bnnPython(m, 1, 0.05) })),
  { name: 'bnn-exact-weakprior-lownoise', code: () => bnnPython('exact', 0.1, 0.01) },
  { name: 'bnn-dropout-weakprior-lownoise', code: () => bnnPython('dropout', 0.1, 0.01) },
  { name: 'bnn-variational-strongprior-highnoise', code: () => bnnPython('variational', 5, 0.25) },
  { name: 'bnn-ensemble-strongprior-highnoise', code: () => bnnPython('ensemble', 5, 0.25) },
  // Gaussian process — every kernel at the defaults (all points), every preset, prior, partial, corners
  ...KERNEL_IDS.map((kid) => ({ name: `gp-${kid}-default`, code: () => gpPython(kid, 0.15, 1, 0.06, 8), timeoutSec: 180 })),
  { name: 'gp-rbf-short', code: () => gpPython('rbf', 0.05, 1, 0.06, 8), timeoutSec: 180 },
  { name: 'gp-periodic-preset', code: () => gpPython('periodic', 0.6, 1, 0.06, 8), timeoutSec: 180 },
  { name: 'gp-rbf-prior', code: () => gpPython('rbf', 0.15, 1, 0.06, 0), timeoutSec: 180 },
  { name: 'gp-linear-prior', code: () => gpPython('linear', 0.15, 1, 0.06, 0), timeoutSec: 180 },
  { name: 'gp-matern-3pts', code: () => gpPython('matern32', 0.15, 1, 0.06, 3), timeoutSec: 180 },
  { name: 'gp-rbf-corner-tight', code: () => gpPython('rbf', 0.03, 2, 0.01, 8), timeoutSec: 180 },
  { name: 'gp-periodic-corner-loose', code: () => gpPython('periodic', 0.6, 0.3, 0.3, 8), timeoutSec: 180 },
  { name: 'gp-rbf-fitted', code: () => gpPython('rbf', 0.27, 0.55, 0.045, 8), timeoutSec: 180 },
  // HMM — the defaults, every preset, slider corners
  { name: 'hmm-default', code: () => hmmPython(0.88, 0.5, 20) },
  { name: 'hmm-sticky-blatant', code: () => hmmPython(0.92, 0.6, 18) },
  { name: 'hmm-subtle-cheat', code: () => hmmPython(0.88, 0.35, 20) },
  { name: 'hmm-twitchy', code: () => hmmPython(0.6, 0.55, 20) },
  { name: 'hmm-long-game', code: () => hmmPython(0.85, 0.5, 24) },
  { name: 'hmm-corner-min', code: () => hmmPython(0.5, 0.17, 10) },
  { name: 'hmm-corner-max', code: () => hmmPython(0.97, 0.8, 24) },
];
