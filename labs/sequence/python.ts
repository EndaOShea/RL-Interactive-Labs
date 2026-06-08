// Runnable NumPy exports for the Sequence-Models labs (template strings — not
// LLM generated). Each mirrors the on-screen maths: the RNN forward pass + the
// BPTT gradient-norm-through-time, the LSTM cell forward + gate/cell dynamics,
// and the seq2seq encoder/decoder forward illustrating the context bottleneck.

/* ---------- 1) RNN — forward pass + BPTT gradient norm ---------- */
export const rnnPython = (
  spectralScale: number,
  seqLen: number,
  hidden = 6,
) => `import numpy as np

# Vanilla RNN — forward pass + backprop-through-time gradient norm.
# Mirrors the lab: hidden size H, sequence length T, recurrent matrix W_hh
# rescaled so its spectral radius (largest singular value) equals SCALE.
H      = ${hidden}
T      = ${seqLen}
SCALE  = ${spectralScale.toFixed(3)}   # spectral radius of W_hh: <1 vanishes, >1 explodes
rng = np.random.default_rng(1)

def spectral_rescale(M, target):
    s = np.linalg.svd(M, compute_uv=False)[0]   # largest singular value
    return M * (target / s)

W_hh = spectral_rescale(rng.uniform(-1, 1, (H, H)), SCALE)
W_xh = rng.uniform(-1, 1, (H, 1)) * 0.6
b    = rng.uniform(-1, 1, H) * 0.3

def forward(x_seq):
    # x_seq: (T,) scalar inputs (e.g. a sine-wave next-step signal)
    h = np.zeros(H)
    H_hist, pre_hist = [], []
    for t in range(len(x_seq)):
        pre = W_hh @ h + W_xh[:, 0] * x_seq[t] + b
        h = np.tanh(pre)
        H_hist.append(h.copy()); pre_hist.append(pre.copy())
    return np.array(H_hist), np.array(pre_hist)

def bptt_grad_norm(pre_hist):
    # ||d h_T / d h_{T-k}|| = || prod_j diag(tanh'(pre_j)) @ W_hh ||
    T = len(pre_hist)
    J = np.eye(H)
    norms = [1.0]
    for k in range(1, T):
        Dp = np.diag(1 - np.tanh(pre_hist[T - k]) ** 2)   # tanh' in (0,1]
        J = J @ (Dp @ W_hh)
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)

if __name__ == "__main__":
    x = np.sin(np.linspace(0, 3 * np.pi, T))
    H_hist, pre_hist = forward(x)
    g = bptt_grad_norm(pre_hist)
    print("hidden state shape :", H_hist.shape)
    print("grad norm vs lag   :", np.round(g, 4))
    print(f"grad @ furthest lag (k={T-1}) = {g[-1]:.3e}")
    print("-> <1 vanishes (long-range memory lost); >1 explodes. Clip or use an LSTM.")
`;

/* ---------- 2) LSTM — cell forward + gate/cell dynamics + grad highway ---------- */
export const lstmPython = (
  gapLen: number,
  forgetBias: number,
  hidden = 6,
) => `import numpy as np

# LSTM cell — forward pass over a sequence with a long-range dependency, plus the
# cell-state gradient "highway". Mirrors the lab: a value is injected early and
# must be carried across a gap of length GAP to the output.
H        = ${hidden}
GAP      = ${gapLen}
F_BIAS   = ${forgetBias.toFixed(2)}    # forget-gate bias: push f -> 1 to open the carousel
T        = GAP + 2
rng = np.random.default_rng(21)

def W(seed, mag): return rng.uniform(-1, 1, (H, 1 + H)) * mag  # input is scalar + hidden
Wf, Wi, Wo, Wg = W(1, .4), W(2, .4), W(3, .4), W(4, .6)
bf, bi, bo, bg = (rng.uniform(-1, 1, H) * .2 for _ in range(4))
sig = lambda z: 1 / (1 + np.exp(-z))

def step(c, h, x):
    xh = np.concatenate(([x], h))
    f = sig(Wf @ xh + bf + F_BIAS)   # forget gate (bias opens the highway)
    i = sig(Wi @ xh + bi)            # input gate
    o = sig(Wo @ xh + bo)            # output gate
    g = np.tanh(Wg @ xh + bg)        # candidate
    c = f * c + i * g                # constant error carousel
    h = o * np.tanh(c)
    return c, h, dict(f=f, i=i, o=o, g=g)

def run(x_seq):
    c, h = np.zeros(H), np.zeros(H)
    F = []
    for x in x_seq:
        c, h, gates = step(c, h, x); F.append(gates['f'])
    return c, h, np.array(F)

def cell_grad_norm(F):
    # d c_T / d c_{T-k}  ~  prod diag(f)  ->  ~ fbar^k  (flat when f ~ 1)
    fbar = F.mean()
    return np.array([fbar ** k for k in range(len(F))])

if __name__ == "__main__":
    x = np.zeros(T); x[0] = 1.0        # inject a value at the start
    c, h, F = run(x)
    print("mean forget gate f :", round(float(F.mean()), 3))
    print("cell grad vs lag   :", np.round(cell_grad_norm(F), 4))
    print("-> f~1 keeps d c_T/d c_0 ~ 1 across the gap; the RNN would have decayed to ~0.")
`;

/* ---------- 3) seq2seq — encoder/decoder + context bottleneck ---------- */
export const seq2seqPython = (
  inputLen: number,
  contextDim: number,
  vocab = 8,
) => `import numpy as np

# seq2seq encoder -> fixed context vector -> decoder, with the context BOTTLENECK
# made explicit. This is an ANALYTIC illustration of capacity vs demand, not a
# trained model: a single DIM-dim vector has finite capacity, while the input
# carries L * log2(V) bits — so early tokens fade as L grows for fixed DIM.
L     = ${inputLen}     # input sequence length
DIM   = ${contextDim}   # context-vector dimension (the bottleneck width)
V     = ${vocab}        # vocabulary size
BITS_PER_DIM = 2.2      # effective bits a real coordinate reliably carries

def encode(tokens, H=DIM):
    # toy encoder RNN: context = last hidden state (a single fixed vector)
    rng = np.random.default_rng(7)
    Whh = rng.uniform(-1, 1, (H, H)) * 0.5
    Wxh = rng.uniform(-1, 1, (H, V)) * 0.6
    h = np.zeros(H)
    for tok in tokens:
        x = np.eye(V)[tok]
        h = np.tanh(Whh @ h + Wxh @ x)
    return h               # <-- the fixed-width context vector

def reconstruction_accuracy(pos, L, dim, vocab):
    capacity = dim * BITS_PER_DIM
    demand   = L * np.log2(vocab)
    budget   = capacity / max(1e-6, demand)
    recency  = pos / (L - 1) if L > 1 else 1.0      # later tokens remembered better
    eff      = budget * (0.55 + 0.9 * recency)
    chance   = 1.0 / vocab
    acc      = 1.0 / (1.0 + np.exp(-3.0 * np.log(eff)))
    return chance + (1 - chance) * acc

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, V, size=L)
    context = encode(tokens)
    print("context vector dim :", context.shape[0], "(everything squeezed through this)")
    print("capacity ~", round(DIM * BITS_PER_DIM, 1), "bits   demand ~", round(L * np.log2(V), 1), "bits")
    acc = [reconstruction_accuracy(p, L, DIM, V) for p in range(L)]
    print("per-position recon :", np.round(acc, 3))
    print(f"first-token acc {acc[0]:.2f}  vs  last-token acc {acc[-1]:.2f}")
    print("-> the single vector forgets the START of long inputs.")
    print("   ATTENTION fixes this: the decoder reads ALL encoder states, not one vector.")
`;
