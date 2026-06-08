// Runnable Python exports for the Stochastic & Bayesian Models labs (template
// strings — not LLM-generated), mirroring each lab's maths and parameters.

export type BnnMode = 'point' | 'dropout' | 'ensemble' | 'variational';

export const bnnPython = (mode: BnnMode, M: number, alpha: number, beta: number) => `import numpy as np

# Bayesian neural network as a fixed random-feature layer + Bayesian linear
# output (mirrors the lab; mode = ${mode}). phi(x) = tanh(w1*x + b1).
rng = np.random.default_rng(0)
M, ALPHA, BETA = ${M}, ${alpha}, ${beta}     # features, prior precision, noise precision
w1 = rng.standard_normal(M) * 4.5
b1 = (rng.random(M) * 2 - 1) * 3

def phi(x):                       # design row for a scalar x
    return np.tanh(w1 * x + b1)

# Training data with a GAP, so uncertainty grows where there is no data.
Xtr = np.r_[np.linspace(0.05, 0.35, 7), np.linspace(0.62, 0.95, 7)]
f = lambda x: 0.8 * np.sin(2 * np.pi * 1.3 * x)
ytr = f(Xtr) + rng.standard_normal(Xtr.size) * 0.05

Phi = np.stack([phi(x) for x in Xtr])             # n x M
# Bayesian linear regression posterior over output weights:
S = np.linalg.inv(ALPHA * np.eye(M) + BETA * Phi.T @ Phi)   # covariance
m = BETA * S @ Phi.T @ ytr                                   # mean

xs = np.linspace(0, 1, 200)
Phis = np.stack([phi(x) for x in xs])
mean = Phis @ m
var = np.einsum('ij,jk,ik->i', Phis, S, Phis) + 1.0 / BETA  # predictive variance
std = np.sqrt(var)

# Mode-specific samples (the "spaghetti" the lab draws):
if "${mode}" == "variational":
    W = rng.multivariate_normal(m, S, size=24)         # sample weights ~ N(m,S)
    samples = Phis @ W.T
elif "${mode}" == "dropout":
    p = 0.2
    samples = np.stack([Phis * (rng.random(M) > p) / (1 - p) @ m for _ in range(24)], axis=1)
elif "${mode}" == "ensemble":
    cols = []
    for k in range(8):
        r = np.random.default_rng(100 + k)
        w1k, b1k = r.standard_normal(M) * 4.5, (r.random(M) * 2 - 1) * 3
        Pk = np.stack([np.tanh(w1k * x + b1k) for x in Xtr])
        Sk = np.linalg.inv(ALPHA * np.eye(M) + BETA * Pk.T @ Pk)
        mk = BETA * Sk @ Pk.T @ ytr
        Ps = np.stack([np.tanh(w1k * x + b1k) for x in xs])
        cols.append(Ps @ mk)
    samples = np.stack(cols, axis=1)
else:  # point estimate — no uncertainty
    samples = mean[:, None]

print("predictive std in the gap (x~0.5):", round(float(std[xs.searchsorted(0.5)]), 3))
print("predictive std at the data (x~0.2):", round(float(std[xs.searchsorted(0.2)]), 3))
`;

export type KernelId = 'rbf' | 'matern32' | 'periodic' | 'linear';

export const gpPython = (k: KernelId, ell: number, sf: number, sn: number) => `import numpy as np

# Gaussian-process regression (mirrors the lab; kernel = ${k}).
ELL, SF, SN = ${ell}, ${sf}, ${sn}          # lengthscale, signal std, noise std
PERIOD = 0.3

def kern(a, b):
    r = np.abs(a[:, None] - b[None, :])
    v = SF ** 2
    if "${k}" == "rbf":      return v * np.exp(-r ** 2 / (2 * ELL ** 2))
    if "${k}" == "matern32": x = np.sqrt(3) * r / ELL; return v * (1 + x) * np.exp(-x)
    if "${k}" == "periodic": s = np.sin(np.pi * r / PERIOD); return v * np.exp(-2 * s ** 2 / ELL ** 2)
    return v * ((a[:, None] - 0.5) * (b[None, :] - 0.5)) + 0.02 * v          # linear

# Data with a gap; the posterior band balloons where there is none.
X = np.r_[np.linspace(0.08, 0.33, 5), np.linspace(0.66, 0.93, 4)]
f = lambda x: np.sin(2 * np.pi * x) * 0.7
y = f(X) + np.random.default_rng(1).standard_normal(X.size) * SN

K = kern(X, X) + SN ** 2 * np.eye(X.size)
Kinv = np.linalg.inv(K)
xs = np.linspace(0, 1, 200)
Ks = kern(xs, X)                 # cross-covariance test x train
mean = Ks @ Kinv @ y
var = np.diag(kern(xs, xs)) - np.einsum('ij,jk,ik->i', Ks, Kinv, Ks)
std = np.sqrt(np.clip(var, 0, None))

# Draw a few posterior sample functions:
cov = kern(xs, xs) - Ks @ Kinv @ Ks.T + 1e-8 * np.eye(xs.size)
L = np.linalg.cholesky(cov + 1e-6 * np.eye(xs.size))
samples = mean[:, None] + L @ np.random.default_rng(2).standard_normal((xs.size, 5))

print("posterior std at a data point :", round(float(std[xs.searchsorted(0.2)]), 3))
print("posterior std in the gap (0.5):", round(float(std[xs.searchsorted(0.5)]), 3))
`;

export const hmmPython = (selfStay: number, loadedSix: number, length: number) => `import numpy as np

# Hidden Markov model — the occasionally-dishonest casino (mirrors the lab).
# Two hidden states: 0 = Fair die, 1 = Loaded die. Observations are die rolls 1..6.
STAY = ${selfStay.toFixed(2)}                 # P(stay in the same die)
A = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])
pi = np.array([0.5, 0.5])

fair = np.full(6, 1 / 6)
p6 = ${loadedSix.toFixed(2)}                   # P(roll a 6) on the loaded die
loaded = np.r_[np.full(5, (1 - p6) / 5), p6]
B = np.stack([fair, loaded])                   # 2 x 6 emission matrix

# Generate a sequence from the true model.
rng = np.random.default_rng(3)
T = ${length}
states, obs = [], []
s = rng.choice(2, p=pi)
for _ in range(T):
    states.append(s)
    obs.append(rng.choice(6, p=B[s]))
    s = rng.choice(2, p=A[s])
obs = np.array(obs)

def forward_backward(obs):
    T = obs.size
    alpha = np.zeros((T, 2)); c = np.zeros(T)          # scaled forward
    alpha[0] = pi * B[:, obs[0]]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
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

post = forward_backward(obs)
path = viterbi(obs)
acc = np.mean(np.array(path) == np.array(states))
print("Viterbi state-recovery accuracy:", round(float(acc), 3))
print("P(loaded | all rolls) at t=0..5:", np.round(post[:6, 1], 2))
`;
