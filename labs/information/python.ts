// Runnable Python exports for the Information Theory labs (template strings —
// not LLM generated), mirroring the on-screen formulas and parameters.

export type LogBase = 'bits' | 'nats';

const baseExpr = (base: LogBase) => (base === 'bits' ? 'np.log2' : 'np.log');
const baseName = (base: LogBase) => (base === 'bits' ? 'bits (log base 2)' : 'nats (natural log)');

// ── 1) Entropy & surprise ──────────────────────────────────────────────────
export const entropyPython = (probs: number[], base: LogBase = 'bits') => {
  const p = probs.map((v) => +v.toFixed(6)).join(', ');
  const log = baseExpr(base);
  return `import numpy as np

# Entropy & surprise — mirrors the lab (base = ${baseName(base)})
# surprise(x) = -log p(x)          (how unexpected an outcome is)
# H(p)        = E[surprise] = -sum p * log p   (average surprise)
p = np.array([${p}])
p = p / p.sum()                    # normalise to a valid distribution

log = ${log}                        # use log2 for bits, np.log for nats

def surprise(p):
    # -log p, with 0*log0 -> 0 handled by masking
    s = np.zeros_like(p)
    nz = p > 0
    s[nz] = -log(p[nz])
    return s

def entropy(p):
    nz = p > 0
    return float(-(p[nz] * log(p[nz])).sum())

H = entropy(p)
H_max = log(len(p))                # uniform distribution = maximum entropy

if __name__ == "__main__":
    print("p          =", np.round(p, 4))
    print("surprise   =", np.round(surprise(p), 4), "(${base})")
    print(f"H(p)       = {H:.4f} ${base}")
    print(f"H_max      = {float(H_max):.4f} ${base}  (uniform)")
    print(f"efficiency = {H / float(H_max):.3f}  (H / H_max)")

    # Draw symbols: the running average surprise of samples -> H(p)
    rng = np.random.default_rng(0)
    draws = rng.choice(len(p), size=10000, p=p)
    avg = np.mean([-float(log(p[i])) for i in draws])
    print(f"sampled avg surprise over 10k draws = {avg:.4f}  (-> H)")
`;
};

// ── 2) KL divergence & cross-entropy ───────────────────────────────────────
export const klPython = (pTrue: number[], qModel: number[], base: LogBase = 'bits') => {
  const p = pTrue.map((v) => +v.toFixed(6)).join(', ');
  const q = qModel.map((v) => +v.toFixed(6)).join(', ');
  const log = baseExpr(base);
  return `import numpy as np

# KL divergence & cross-entropy — mirrors the lab (base = ${baseName(base)})
# H(p)    = -sum p log p                  (entropy of the truth)
# H(p,q)  = -sum p log q                  (cross-entropy = the classification loss)
# KL(p||q)= sum p log(p/q) = H(p,q) - H(p)   (extra bits from coding p with q)
p = np.array([${p}]);  p = p / p.sum()      # true distribution
q = np.array([${q}]);  q = q / q.sum()      # model distribution

log = ${log}
EPS = 1e-12

def entropy(p):
    nz = p > 0
    return float(-(p[nz] * log(p[nz])).sum())

def cross_entropy(p, q):
    nz = p > 0
    return float(-(p[nz] * log(q[nz] + EPS)).sum())

def kl(p, q):
    nz = p > 0
    return float((p[nz] * log((p[nz]) / (q[nz] + EPS))).sum())

# ── train q -> p by gradient descent on cross-entropy w.r.t. q's logits ──
# q = softmax(z). d/dz H(p,q) = q - p, so we step z <- z - lr*(q - p).
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def fit_q_to_p(p, steps=400, lr=0.5):
    z = np.zeros_like(p)
    for _ in range(steps):
        qz = softmax(z)
        z -= lr * (qz - p)          # gradient of cross-entropy on the logits
    return softmax(z)

if __name__ == "__main__":
    print(f"H(p)      = {entropy(p):.4f} ${base}")
    print(f"H(p,q)    = {cross_entropy(p, q):.4f} ${base}")
    print(f"KL(p||q)  = {kl(p, q):.4f} ${base}")
    print(f"KL(q||p)  = {kl(q, p):.4f} ${base}   (asymmetric: KL(p||q) != KL(q||p))")

    q_star = fit_q_to_p(p)
    print("\\nafter minimising cross-entropy (q -> p):")
    print("q*        =", np.round(q_star, 4))
    print(f"H(p,q*)   = {cross_entropy(p, q_star):.4f}  -> floor is H(p) = {entropy(p):.4f}")
    print(f"KL(p||q*) = {kl(p, q_star):.4f}  -> 0")
`;
};

// ── 3) Source coding — Huffman & the entropy limit ─────────────────────────
export const huffmanPython = (symbols: string[], probs: number[], base: LogBase = 'bits') => {
  const syms = symbols.map((s) => `'${s}'`).join(', ');
  const p = probs.map((v) => +v.toFixed(6)).join(', ');
  const log = baseExpr(base);
  return `import numpy as np
import heapq
from math import ceil, log2

# Source coding: Huffman codes vs the entropy bound — mirrors the lab.
# Shannon source-coding theorem:  H(p) <= L < H(p) + 1   (per symbol)
# Huffman is the OPTIMAL prefix code: it minimises L = sum p * len(code).
symbols = [${syms}]
p = np.array([${p}]);  p = p / p.sum()

log = ${log}

def entropy(p):
    nz = p > 0
    return float(-(p[nz] * log(p[nz])).sum())

def huffman(symbols, probs):
    # Repeatedly merge the two least-probable nodes (a min-heap by probability).
    heap = [[w, i, sym] for i, (w, sym) in enumerate(zip(probs, symbols))]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)        # least probable
        hi = heapq.heappop(heap)        # next least probable
        # children stored as [prob, tie, (left_node, right_node)]
        heapq.heappush(heap, [lo[0] + hi[0], tie, (lo, hi)]); tie += 1
    root = heap[0]

    codes = {}
    def walk(node, prefix=""):
        payload = node[2]
        if isinstance(payload, tuple):  # internal node -> recurse (0 left, 1 right)
            walk(payload[0], prefix + "0")
            walk(payload[1], prefix + "1")
        else:                           # leaf -> assign the accumulated bits
            codes[payload] = prefix or "0"
    walk(root)
    return codes

if __name__ == "__main__":
    codes = huffman(symbols, list(p))
    L = sum(pi * len(codes[s]) for pi, s in zip(p, symbols))   # average code length
    H = entropy(p)
    fixed = ceil(log2(len(symbols)))                            # fixed-length bits/symbol
    for s, pi in sorted(zip(symbols, p), key=lambda t: -t[1]):
        print(f"  {s!r:>5}  p={pi:.3f}  code={codes[s]:>6}  len={len(codes[s])}")
    print(f"\\nH(p)        = {H:.4f} ${base}   (entropy = lower bound on L)")
    print(f"avg length L = {L:.4f} ${base}/symbol   (Huffman, optimal prefix code)")
    print(f"efficiency   = {H / L:.3f}  (H / L; -> 1 is optimal)")
    print(f"fixed-length = {fixed} bits/symbol  ->  Huffman saves {fixed - L:.3f} bits/symbol")
`;
};
