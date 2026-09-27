// Runnable NumPy exports for the Sequence-Models labs (template strings — not
// LLM generated). Each mirrors its lab exactly: the same seeded weights (a port
// of the labs' mulberry32 PRNG), the same inputs, the same exact Jacobian /
// probe computations and the same verdict thresholds as labs/sequence/shared.ts.
import type { PythonSample } from '../../utils/pythonSamples';
import { REGIME_BAND } from './shared';

const num = (v: number, d = 3) => v.toFixed(d);

// Shared Python prelude: the PRNG port + the seeded initialisers.
const PRELUDE = `import numpy as np


def mulberry32(seed):
    """Exact port of the lab's seeded PRNG (labs/sequence/shared.ts): same seed,
    same numbers, same order — so every weight below equals the one on screen."""
    a = seed & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def uniform_matrix(rows, cols, seed, mag=1.0):
    r = mulberry32(seed)                      # uniform [-1, 1) * mag, filled row by row
    return np.array([[(r() * 2 - 1) * mag for _ in range(cols)] for _ in range(rows)])


def bias_vector(n, seed, mag):
    r = mulberry32(seed)
    return np.array([(r() * 2 - 1) * mag for _ in range(n)])
`;

// Scaled orthogonal W_hh (modified Gram–Schmidt on the seeded rows) — the RNN
// lab's matrix and the LSTM lab's RNN baseline.
const ORTHO = `

def orthogonal_recurrent(h, rho, seed):
    """W_hh = rho * Q, Q orthogonal (modified Gram-Schmidt on the seeded rows):
    every singular value is rho and every eigenvalue has modulus rho."""
    Q = []
    for row in uniform_matrix(h, h, seed):
        v = row.copy()
        for q in Q:
            v = v - np.dot(v, q) * q
        Q.append(v / np.sqrt(np.dot(v, v)))
    return np.array(Q) * rho


def rnn_jacobian_norms(states, W_hh):
    """Exact ||d h_{t-1} / d h_{t-1-k}||_2 = || prod_m diag(1 - h_{t-1-m}^2) W_hh ||_2."""
    t = len(states)
    J = np.eye(W_hh.shape[0])
    norms = [1.0]
    for k in range(1, t):
        h = states[t - k]
        J = J @ (np.diag(1 - h ** 2) @ W_hh)
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)
`;

/* ---------- 1) RNN — forward pass + exact BPTT Jacobian norms ---------- */
export const rnnPython = (rho: number, seqLen: number, hidden = 6) => `${PRELUDE}${ORTHO}
# Vanilla RNN — forward pass + EXACT backprop-through-time Jacobian norms.
# Mirrors the lab: hidden size H, sequence length T, and W_hh = RHO * Q with Q
# orthogonal, so RHO is both the spectral radius and the largest singular value.
H    = ${hidden}
T    = ${seqLen}
RHO  = ${num(rho, 2)}
BAND = ${REGIME_BAND}          # per-step factor within 1 +/- BAND counts as near-critical

W_hh = orthogonal_recurrent(H, RHO, seed=1)
W_xh = uniform_matrix(H, 1, seed=7, mag=0.6)
b    = bias_vector(H, seed=13, mag=0.3)


def forward(xs):
    h = np.zeros(H)
    states = []
    for x_t in xs:
        h = np.tanh(W_hh @ h + W_xh[:, 0] * x_t + b)
        states.append(h)
    return states


def classify(norms):
    K = len(norms) - 1
    per_step = norms[-1] ** (1 / K) if norms[-1] > 0 else 0.0   # geometric-mean factor per step
    regime = "vanishing" if per_step < 1 - BAND else "exploding" if per_step > 1 + BAND else "near-critical"
    return per_step, regime


if __name__ == "__main__":
    x = np.sin((np.arange(T) / max(1, T - 1)) * 3 * np.pi)      # the lab's sine input
    states = forward(x)
    norms = rnn_jacobian_norms(states, W_hh)
    # sub-multiplicative bound: ||J_k|| <= prod_m max_i(1 - h^2) * ||W_hh||_2^k
    bound = np.cumprod([1.0] + [np.max(1 - states[T - k] ** 2) * np.linalg.norm(W_hh, 2) for k in range(1, T)])
    per_step, regime = classify(norms)
    print("spectral radius  :", round(float(np.max(np.abs(np.linalg.eigvals(W_hh)))), 4),
          "  largest singular value:", round(float(np.linalg.norm(W_hh, 2)), 4))
    print("mean tanh'(1-h^2):", round(float(np.mean([1 - s ** 2 for s in states])), 4))
    print("||dh_T/dh_(T-k)|| :", np.array2string(norms, precision=4))
    print("upper bound       :", np.array2string(bound, precision=4))
    print(f"furthest lag k={T - 1}: {norms[-1]:.3e}   per-step factor {per_step:.3f}  ->  {regime}")
`;

/* ---------- 2) LSTM — gates, direct cell path, carry across the gap ---------- */
export const lstmPython = (gap: number, forgetBias: number, hidden = 6, rnnRho = 1) => `${PRELUDE}${ORTHO}
# LSTM cell carrying an injected value across a gap, vs a vanilla RNN.
# Mirrors the lab: the same seeded gate weights, input x = [1, 0, 0, ...] of
# length GAP + 2, the direct cell-path Jacobian prod_j diag(f_j), and the
# measured influence of the injected value (run with x_1 = 1 minus run with x_1 = 0).
H      = ${hidden}
GAP    = ${gap}
F_BIAS = ${num(forgetBias, 2)}      # added to the forget pre-activation: pushes f -> 1
T      = GAP + 2
RNN_RHO = ${num(rnnRho, 2)}     # the vanilla-RNN baseline: W_hh = RNN_RHO * Q (orthogonal)

Wf, Wi = uniform_matrix(H, 1 + H, 21, 0.4), uniform_matrix(H, 1 + H, 22, 0.4)
Wo, Wg = uniform_matrix(H, 1 + H, 23, 0.4), uniform_matrix(H, 1 + H, 24, 0.6)
bf, bi = bias_vector(H, 31, 0.2), bias_vector(H, 32, 0.2)
bo, bg = bias_vector(H, 33, 0.2), bias_vector(H, 34, 0.2)
sig = lambda z: 1 / (1 + np.exp(-z))


def run_lstm(xs):
    c, h = np.zeros(H), np.zeros(H)
    C, Hs, F = [], [], []
    for x in xs:
        xh = np.concatenate(([x], h))            # [x, h_prev]
        f = sig(Wf @ xh + bf + F_BIAS)            # forget gate
        i = sig(Wi @ xh + bi)                     # input gate
        o = sig(Wo @ xh + bo)                     # output gate
        g = np.tanh(Wg @ xh + bg)                 # candidate
        c = f * c + i * g                         # additive carry (constant error carousel)
        h = o * np.tanh(c)
        C.append(c); Hs.append(h); F.append(f)
    return np.array(C), np.array(Hs), np.array(F)


W_hh = orthogonal_recurrent(H, RNN_RHO, seed=1)
U    = uniform_matrix(H, 1, seed=7, mag=0.6)
b    = bias_vector(H, seed=13, mag=0.3)


def run_rnn(xs):
    h, out = np.zeros(H), []
    for x in xs:
        h = np.tanh(W_hh @ h + U[:, 0] * x + b)
        out.append(h)
    return out


def verdict(v, hi, lo, names):
    return names[0] if v >= hi else names[1] if v >= lo else names[2]


if __name__ == "__main__":
    x = np.zeros(T); x[0] = 1.0                   # inject a value, then silence
    C, Hs, F = run_lstm(x)
    C0, H0, _ = run_lstm(np.zeros(T))             # same cell, nothing injected
    # direct cell path d c_T / d c_(T-k) along the carry: prod of diag(f) -> per-unit products
    prod = np.cumprod(F[::-1][:T - 1], axis=0)    # rows k = 1..T-1
    path_max = np.concatenate(([1.0], prod.max(axis=1)))
    path_mean = np.concatenate(([1.0], prod.mean(axis=1)))
    R, R0 = run_rnn(x), run_rnn(np.zeros(T))
    rnn = rnn_jacobian_norms(R, W_hh)
    d_c = np.linalg.norm(C - C0, axis=1)
    d_h = np.linalg.norm(Hs - H0, axis=1)
    d_r = np.array([np.linalg.norm(a - z) for a, z in zip(R, R0)])
    print("mean forget gate f       :", round(float(F.mean()), 4))
    print("cell path max|mean (lag) :", np.array2string(path_max, precision=4), np.array2string(path_mean, precision=4))
    print("RNN ||dh_T/dh_(T-k)||    :", np.array2string(rnn, precision=4))
    print(f"lag {T - 1}: LSTM cell path {path_max[-1]:.3e} ({verdict(path_max[-1], 0.5, 0.01, ('open', 'leaky', 'closed'))})"
          f"   RNN {rnn[-1]:.3e}")
    for name, tr in (("LSTM |dh_t|", d_h), ("LSTM |dc_t|", d_c), ("RNN  |dh_t|", d_r)):
        print(f"{name}: {np.array2string(tr, precision=4)}")
    for name, tr in (("LSTM", d_h), ("RNN", d_r)):
        ratio = tr[-1] / tr[0]
        print(f"{name} carry |dh_T|/|dh_1| = {ratio:.3e} -> {verdict(ratio, 0.5, 0.01, ('carried', 'faded', 'lost'))}")
`;

/* ---------- 3) seq2seq — encoder, context vector, measured bottleneck ---------- */
export const seq2seqPython = (inputLen: number, contextDim: number, vocab: number, opts: { nTrain: number; nTest: number; seed: number; ridge: number }) => `${PRELUDE}
# seq2seq context bottleneck — a real measurement. An (untrained, seeded) RNN
# encoder squeezes a token sequence into its last state h_L (the context). A
# linear probe (ridge least squares) per position tries to read each token back
# from h_L alone, versus from that position's own state h_p (attention with the
# ideal alignment). Mirrors the lab's encoder, samples, probe and scoring.
L, DIM, V = ${inputLen}, ${contextDim}, ${vocab}
N_TRAIN, N_TEST, SEED, RIDGE = ${opts.nTrain}, ${opts.nTest}, ${opts.seed}, ${opts.ridge}

M = uniform_matrix(DIM, DIM, 5)
W_hh = M * (0.9 / np.linalg.norm(M, 2))       # largest singular value = 0.9
W_xh = uniform_matrix(DIM, V, 6, 0.6)
b = bias_vector(DIM, 9, 0.2)


def encode(tokens):
    h, states = np.zeros(DIM), []
    for tok in tokens:
        h = np.tanh(W_hh @ h + W_xh @ np.eye(V)[tok] + b)
        states.append(h)
    return np.array(states)                    # states[-1] is the context vector


def fit(F, tokens):                            # (F^T F + RIDGE I)^-1 F^T Y
    Y = np.eye(V)[tokens]
    return np.linalg.solve(F.T @ F + RIDGE * np.eye(F.shape[1]), F.T @ Y)


def sensitivities(states):                     # ||d h_L / d x_p||_2 for every p
    out, G = np.zeros(L), np.eye(DIM)
    for p in range(L - 1, -1, -1):
        D = np.diag(1 - states[p] ** 2)
        out[p] = np.linalg.norm(G @ D @ W_xh, 2)
        G = G @ D @ W_hh
    return out


if __name__ == "__main__":
    r = mulberry32(SEED)
    draw = lambda n: np.array([[int(r() * V) for _ in range(L)] for _ in range(n)])
    train, test = draw(N_TRAIN), draw(N_TEST)
    Htr = np.array([encode(s) for s in train])
    Hte = np.array([encode(s) for s in test])
    feat = lambda X: np.hstack([X, np.ones((len(X), 1))])
    ctx, att, demo_ctx, demo_att = [], [], [], []
    for p in range(L):
        Wc = fit(feat(Htr[:, L - 1]), train[:, p])
        Wa = fit(feat(Htr[:, p]), train[:, p])
        pc = (feat(Hte[:, L - 1]) @ Wc).argmax(axis=1)
        pa = (feat(Hte[:, p]) @ Wa).argmax(axis=1)
        ctx.append(float(np.mean(pc == test[:, p])))
        att.append(float(np.mean(pa == test[:, p])))
        demo_ctx.append(int(pc[0])); demo_att.append(int(pa[0]))
    ctx, att = np.array(ctx), np.array(att)
    sens = np.mean([sensitivities(h) for h in Hte], axis=0)
    chance = 1 / V
    print("context h_L probe accuracy :", np.array2string(ctx, precision=3))
    print("attention (h_p) accuracy   :", np.array2string(att, precision=3))
    print("mean ||dh_L/dx_p||         :", np.array2string(sens, precision=4))
    print(f"mean accuracy: context {ctx.mean():.3f}  vs  attention {att.mean():.3f}   (chance {chance:.3f})")
    print(f"tokens recovered from h_L ~ {np.sum((ctx - chance) / (1 - chance)):.2f} of {L}")
    print("demo (1st held-out) sequence :", test[0])
    print("  decoded from context       :", np.array(demo_ctx))
    print("  decoded with attention     :", np.array(demo_att))
`;

/* ---------- export check: one entry per export × representative params ---------- */
const S2S_OPTS = { nTrain: 300, nTest: 200, seed: 2024, ridge: 0.001 };

export const PYTHON_SAMPLES: PythonSample[] = [
  // RNN — default, every preset, slider edges
  { name: 'rnn-default', code: () => rnnPython(1.0, 14) },
  { name: 'rnn-vanishing', code: () => rnnPython(0.55, 16) },
  { name: 'rnn-near-critical', code: () => rnnPython(1.5, 14) },
  { name: 'rnn-exploding', code: () => rnnPython(1.8, 16) },
  { name: 'rnn-long-sequence', code: () => rnnPython(1.0, 20) },
  { name: 'rnn-edge-min', code: () => rnnPython(0.4, 6) },
  { name: 'rnn-edge-max', code: () => rnnPython(2.0, 20) },
  // LSTM — default, every preset, slider edges
  { name: 'lstm-default', code: () => lstmPython(12, 3.0) },
  { name: 'lstm-highway-open', code: () => lstmPython(12, 4.0) },
  { name: 'lstm-leaky', code: () => lstmPython(12, 1.0) },
  { name: 'lstm-long-carry', code: () => lstmPython(16, 5.0) },
  { name: 'lstm-closing-gate', code: () => lstmPython(10, -1.5) },
  { name: 'lstm-edge-min', code: () => lstmPython(6, -3.0) },
  { name: 'lstm-edge-max', code: () => lstmPython(18, 5.0) },
  // seq2seq — default, every preset, slider edges
  { name: 's2s-default', code: () => seq2seqPython(12, 4, 4, S2S_OPTS) },
  { name: 's2s-roomy', code: () => seq2seqPython(3, 10, 4, S2S_OPTS) },
  { name: 's2s-long-input', code: () => seq2seqPython(18, 12, 4, S2S_OPTS) },
  { name: 's2s-tiny', code: () => seq2seqPython(10, 2, 4, S2S_OPTS) },
  { name: 's2s-edge-min', code: () => seq2seqPython(3, 2, 4, S2S_OPTS) },
  { name: 's2s-edge-max', code: () => seq2seqPython(18, 12, 4, S2S_OPTS), timeoutSec: 120 },
];
