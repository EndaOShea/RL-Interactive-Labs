// Runnable Python exports for the Information Theory labs (template strings —
// not LLM generated). Each mirrors its lab exactly: the lab's own distributions
// (full precision), the same conventions (H_max over the support; exact ∞ for
// log 0; logits floored at 1e-9; lr 0.5; stop at KL < 1e-4 in the chosen unit) and
// the same Huffman tie-breaking (probability, then creation order).
import type { PythonSample } from '../../utils/pythonSamples';
import type { LogBase } from './infoMath';

export type { LogBase } from './infoMath';

const baseExpr = (base: LogBase) => (base === 'bits' ? 'np.log2' : 'np.log');
const baseName = (base: LogBase) => (base === 'bits' ? 'bits (log base 2)' : 'nats (natural log)');
/** A JS number as a Python literal (shortest round-trip repr — the same double in both). */
const num = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');
const arr = (xs: number[]) => xs.map(num).join(', ');

// ── 1) Entropy & surprise ──────────────────────────────────────────────────
export const entropyPython = (probs: number[], base: LogBase = 'bits') => {
  const log = baseExpr(base);
  return `import numpy as np

# Entropy & surprise — mirrors the lab (base = ${baseName(base)})
# surprise(x) = -log p(x)          (how unexpected an outcome is)
# H(p)        = E[surprise] = -sum p * log p   (average surprise)
# H_max       = log N, N = number of outcomes with p > 0 (the uniform law on the support)
p = np.array([${arr(probs)}])   # the lab's distribution, full precision

log = ${log}                        # log2 for bits, np.log for nats

def surprise(p):
    # -log p, with 0*log0 -> 0 handled by masking
    s = np.zeros_like(p)
    nz = p > 0
    s[nz] = -log(p[nz])
    return s

def entropy(p):
    nz = p > 0
    return float(-(p[nz] * log(p[nz])).sum()) + 0.0   # (+ 0.0 turns -0.0 into 0.0)

H = entropy(p)
N = int(np.count_nonzero(p))       # live outcomes (the support)
H_max = float(log(N)) if N > 1 else 0.0

if __name__ == "__main__":
    print("p          =", np.round(p, 4))
    print("surprise   =", np.round(surprise(p), 4), "(${base}; 0 shown for impossible outcomes)")
    print(f"H(p)       = {H:.4f} ${base}")
    print(f"H_max      = log {N} = {H_max:.4f} ${base}  (uniform over the {N} live outcomes)")
    print("efficiency = " + (f"{H / H_max:.3f}  (H / H_max)" if H_max > 0 else "undefined (a single live outcome: H = H_max = 0)"))

    # Draw symbols: the running average surprise of samples -> H(p)  (law of large numbers)
    rng = np.random.default_rng(0)
    draws = rng.choice(len(p), size=10000, p=p / p.sum())
    avg = float(np.mean(-log(p[draws]))) + 0.0
    print(f"sampled avg surprise over 10k draws = {avg:.4f}  (-> H)")
`;
};

// ── 2) KL divergence & cross-entropy ───────────────────────────────────────
export const klPython = (pTrue: number[], qModel: number[], base: LogBase = 'bits') => {
  const log = baseExpr(base);
  return `import numpy as np

# KL divergence & cross-entropy — mirrors the lab (base = ${baseName(base)})
# H(p)     = -sum p log p                    (entropy of the truth)
# H(p,q)   = -sum p log q                    (cross-entropy = the classification loss)
# KL(p||q) = sum p log(p/q) = H(p,q) - H(p)  (extra ${base} from coding p with q's code)
# Terms with p = 0 contribute 0; a term with p > 0 but q = 0 is +infinity (the log-0 trap).
p = np.array([${arr(pTrue)}])   # true distribution (the lab's, full precision)
q = np.array([${arr(qModel)}])   # model distribution (the lab's q at download time)

log = ${log}
UNIT = "${base}"

def entropy(p):
    nz = p > 0
    return float(-(p[nz] * log(p[nz])).sum())

def cross_entropy(p, q):
    nz = p > 0
    if np.any(q[nz] <= 0):
        return float("inf")
    return float(-(p[nz] * log(q[nz])).sum())

def kl(p, q):
    nz = p > 0
    if np.any(q[nz] <= 0):
        return float("inf")
    return float((p[nz] * log(p[nz] / q[nz])).sum())

# ── train q -> p by gradient descent on cross-entropy w.r.t. q's logits ──
# q = softmax(z). dH(p,q)/dz = q - p, so each step is z <- z - lr*(q - p).
# Like the lab: start from z = ln max(q, 1e-9) (a softmax cannot hold an exact 0),
# lr = 0.5, and stop as soon as KL(p||q) < 1e-4 in the chosen unit.
LR, KL_STOP, LOGIT_EPS = 0.5, 1e-4, 1e-9

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def fit_q_to_p(p, q0, max_steps=1_000_000):
    z = np.log(np.maximum(q0, LOGIT_EPS))
    for step in range(1, max_steps + 1):
        z = z - LR * (softmax(z) - p)       # gradient of cross-entropy on the logits
        q = softmax(z)
        if kl(p, q) < KL_STOP:
            return q, step
    return softmax(z), max_steps

def show(x):
    return "inf" if x == float("inf") else f"{x:.4f}"

if __name__ == "__main__":
    print(f"H(p)      = {show(entropy(p))} {UNIT}")
    print(f"H(p,q)    = {show(cross_entropy(p, q))} {UNIT}")
    print(f"KL(p||q)  = {show(kl(p, q))} {UNIT}   (forward: mass-covering)")
    print(f"KL(q||p)  = {show(kl(q, p))} {UNIT}   (reverse: mode-seeking)")
    a, b = kl(p, q), kl(q, p)
    print("asymmetry:", "KL(p||q) = KL(q||p) here" if a == b or abs(a - b) < 5e-4 else "KL(p||q) != KL(q||p)")

    q_star, steps = fit_q_to_p(p, q)
    print(f"\\nafter {steps} gradient steps on the cross-entropy (q -> p):")
    print("q*        =", np.round(q_star, 4))
    print(f"H(p,q*)   = {cross_entropy(p, q_star):.4f}  -> floor is H(p) = {entropy(p):.4f}")
    print(f"KL(p||q*) = {kl(p, q_star):.6f}  (< {KL_STOP})")
`;
};

// ── 3) Source coding — Huffman & the entropy limit ─────────────────────────
export const huffmanPython = (symbols: string[], weights: number[]) => {
  const syms = symbols.map((s) => JSON.stringify(s)).join(', ');
  return `import heapq
from math import ceil, log2

# Source coding: Huffman codes vs the entropy bound — mirrors the lab exactly.
# Shannon's source-coding theorem:  H(p) <= L < H(p) + 1   (bits per symbol)
# Huffman is the OPTIMAL prefix code: it minimises L = sum p * len(code).
symbols = [${syms}]
weights = [${arr(weights)}]           # the lab's symbol weights
total = sum(weights)
p = [w / total for w in weights]   # normalised exactly as in the lab

def entropy(p):
    return -sum(pi * log2(pi) for pi in p if pi > 0)

def huffman(symbols, probs):
    # Repeatedly merge the two least-probable nodes; ties go to the node created
    # first (leaves in symbol order, then parents in merge order). The lower node
    # hangs on the 0-edge, the higher on the 1-edge.
    heap = [[pr, i, sym] for i, (pr, sym) in enumerate(zip(probs, symbols))]
    heapq.heapify(heap)
    next_id = len(heap)
    merges = []
    while len(heap) > 1:
        lo = heapq.heappop(heap)        # least probable
        hi = heapq.heappop(heap)        # next least probable
        merges.append(lo[0] + hi[0])
        heapq.heappush(heap, [lo[0] + hi[0], next_id, (lo, hi)])
        next_id += 1
    codes = {}
    def walk(node, prefix=""):
        payload = node[2]
        if isinstance(payload, tuple):  # internal node -> recurse (0 left, 1 right)
            walk(payload[0], prefix + "0")
            walk(payload[1], prefix + "1")
        else:                           # leaf -> the accumulated bits
            codes[payload] = prefix or "0"
    walk(heap[0])
    return codes, merges

if __name__ == "__main__":
    codes, merges = huffman(symbols, p)
    H = entropy(p)
    L = sum(pi * len(codes[s]) for pi, s in zip(p, symbols))    # average code length
    fixed = ceil(log2(max(2, len(symbols))))                     # fixed-length bits/symbol
    shannon = [ceil(-log2(pi)) for pi in p]                      # Shannon lengths ceil(-log2 p)
    L_shannon = sum(pi * l for pi, l in zip(p, shannon))
    kraft = sum(2.0 ** -len(codes[s]) for s in symbols)
    print("   sym      p   -log2 p   code      len  ceil(-log2 p)")
    for s, pi, sl in sorted(zip(symbols, p, shannon), key=lambda t: -t[1]):
        print(f"  {s!r:>5}  {pi:.3f}  {-log2(pi):7.2f}   {codes[s]:<8}  {len(codes[s]):>3}  {sl:>6}")
    running, acc = [], 0.0
    for m in merges:
        acc += m
        running.append(round(acc, 3))
    print("\\nL so far after each merge (+ the parent's probability):", running)
    print(f"H(p)          = {H:.4f} bits   (entropy = lower bound on L)")
    print(f"avg length L  = {L:.4f} bits/symbol   (Huffman, optimal prefix code; = sum of merge probabilities)")
    print(f"efficiency    = {H / L:.3f}  (H / L)")
    print(f"Kraft sum     = {kraft:.4f}  (sum 2^-len = 1: a full binary code tree)")
    print(f"Shannon code  : L = {L_shannon:.4f} bits with lengths ceil(-log2 p)  (H <= L_Shannon < H + 1, and L_Huffman <= L_Shannon)")
    print(f"fixed-length  = {fixed} bits/symbol  ->  Huffman saves {fixed - L:.3f} bits/symbol")
`;
};

// ---- export-check samples -----------------------------------------------------
const norm = (w: number[]) => { const s = w.reduce((a, b) => a + b, 0); return w.map((v) => v / s); };
const ENTROPY_W: [string, number[]][] = [
  ['fair-die', [1, 1, 1, 1, 1, 1]], ['loaded-die', [4, 1, 1, 1, 1, 1]], ['near-certain', [40, 1, 0.4, 0.4, 0.4, 0.4]],
  ['fair-coin', [1, 1, 0, 0, 0, 0]], ['biased-coin', [4, 1, 0, 0, 0, 0]], ['single-outcome', [0, 0, 7, 0, 0, 0]],
];
const KL_PQ: [string, number[], number[]][] = [
  ['perfect-match', [4, 3, 2, 1, 1], [4, 3, 2, 1, 1]],
  ['over-confident', [3, 3, 2, 1, 1], [9, 1, 0.5, 0.3, 0.3]],
  ['misses-a-mode', [3, 1, 3, 1, 2], [5, 4, 0, 3, 0]],
  ['asymmetry', [1, 1, 1, 1, 1], [10, 0.1, 0.1, 0.1, 0.1]],
  ['label-smoothing', [20, 0.2, 0.2, 0.2, 0.2], [16, 1, 1, 1, 1]],
  ['slider-extremes', [0.1, 10, 0.1, 10, 0.1], [0, 0, 0, 0, 10]],
];
const HUFF: [string, string[], number[]][] = [
  ['english', ['E', 'T', 'A', 'O', 'J', 'Q', 'X', 'Z'], [12.7, 9.1, 8.2, 7.5, 0.2, 0.1, 0.2, 0.1]],
  ['skewed', ['a', 'b', 'c', 'd', 'e', 'f'], [40, 20, 15, 12, 8, 5]],
  ['near-uniform', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [13, 13, 12, 12, 13, 12, 13, 12]],
  ['power-of-half', ['a', 'b', 'c', 'd', 'e'], [16, 8, 4, 2, 2]],
  ['slider-extremes', ['a', 'b', 'c', 'd', 'e', 'f'], [50, 0.1, 0.1, 50, 0.1, 25]],
];
export const PYTHON_SAMPLES: PythonSample[] = [
  ...ENTROPY_W.flatMap(([n, w]) => (['bits', 'nats'] as LogBase[]).map((b) => ({ name: `entropy-${n}-${b}`, code: () => entropyPython(norm(w), b) }))),
  ...KL_PQ.flatMap(([n, p, q]) => (['bits', 'nats'] as LogBase[]).map((b) => ({ name: `kl-${n}-${b}`, code: () => klPython(norm(p), norm(q), b) }))),
  ...HUFF.map(([n, s, w]) => ({ name: `huffman-${n}`, code: () => huffmanPython(s, w) })),
];
