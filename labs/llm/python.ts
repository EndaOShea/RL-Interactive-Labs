// Runnable NumPy exports for the LLM labs — mirror the in-browser toy models.

import { RagParams } from './rag';

// Minimal stub — replaced with a full runnable export in Task 20.
export function ragPython(variantId: string, params: RagParams): string {
  return `# RAG export (${variantId}) — generated in a later task\n`;
}

export const tokenizerPython = (mode: 'greedy' | 'bpe' = 'greedy', merges = 20) => mode === 'bpe' ? `from collections import Counter

# Byte-Pair Encoding — LEARN the merges from a corpus, then apply them.
# Mirrors the lab's "BPE (learn merges)" mode.
CORPUS = "the cat sat on the mat the cat ran fast the dog sat"
NUM_MERGES = ${merges}

def get_stats(words):
    pairs = Counter()
    for w, freq in words.items():
        syms = w.split()
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i + 1])] += freq
    return pairs

def merge_pair(pair, words):
    bigram = " ".join(pair)
    joined = "".join(pair)
    return {w.replace(bigram, joined): f for w, f in words.items()}

if __name__ == "__main__":
    # start from characters: each word is a space-separated char sequence + end marker
    vocab = Counter(CORPUS.split())
    words = {" ".join(list(w)) + " </w>": f for w, f in vocab.items()}

    merges = []
    for step in range(NUM_MERGES):
        stats = get_stats(words)
        if not stats:
            break
        best = max(stats, key=stats.get)              # most frequent adjacent pair
        words = merge_pair(best, words)
        merges.append(best)
        print(f"merge {step + 1:2d}: {best[0]!r}+{best[1]!r} -> {''.join(best)!r}  (count {stats[best]})")

    print("\\nlearned", len(merges), "merges")
    print("final pieces:", sorted({s for w in words for s in w.split()}))
` : `import re

# Deterministic greedy subword tokenizer — mirrors the lab.
# Real tokenizers (BPE) LEARN their merges from data; here they are fixed.
SUFFIXES = ["tion", "ing", "ed", "ly", "ization", "iza", "er", "es", "s"]
PREFIXES = ["token", "trans", "un", "re", "pre"]

def split_word(w):
    pieces = []
    # greedy prefix
    for p in PREFIXES:
        if w.lower().startswith(p) and len(w) > len(p):
            pieces.append(w[:len(p)]); w = w[len(p):]; break
    # greedy suffix(es)
    tail = []
    changed = True
    while changed and len(w) > 3:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if w.lower().endswith(s) and len(w) > len(s):
                tail.insert(0, w[-len(s):]); w = w[:-len(s)]; changed = True; break
    if w:
        pieces.append(w)
    pieces.extend(tail)
    # mark continuation pieces BPE-style
    return [pieces[0]] + ["##" + p for p in pieces[1:]] if pieces else []

def tokenize(text):
    # split on whitespace + keep punctuation as its own token
    raw = re.findall(r"[A-Za-z]+|[0-9]+|[^\\sA-Za-z0-9]", text)
    toks = []
    for w in raw:
        toks.extend(split_word(w) if w.isalpha() and len(w) > 6 else [w])
    return toks

if __name__ == "__main__":
    text = "Tokenization powers transformers."
    toks = tokenize(text)
    vocab = {t: i for i, t in enumerate(sorted(set(toks)))}
    ids = [vocab[t] for t in toks]
    print("tokens:", toks)
    print("ids   :", ids)
    print(f"{len(text)} chars -> {len(toks)} tokens "
          f"({len(text)/max(1,len(toks)):.2f} chars/token)")
`;

export const samplingPython = (
  temp: number, topk: number, topp: number, minp = 0, repPenalty = 1,
) => `import numpy as np

# Toy autoregressive sampler — mirrors the lab.
np.random.seed(0)
VOCAB = ["the", "cat", "sat", "on", "a", "mat", "and", "ran", "fast", ".", "dog", "saw"]

# fixed bigram-ish logits: logits[prev_id] -> scores over next token
rng = np.random.default_rng(42)
LOGITS = rng.normal(0, 1.5, size=(len(VOCAB), len(VOCAB)))

TEMPERATURE = ${temp}
TOP_K = ${topk}
TOP_P = ${topp}
MIN_P = ${minp}            # min-p: drop tokens below MIN_P * p_max (0 = off)
REP_PENALTY = ${repPenalty}  # repetition penalty on already-generated ids (1 = off)

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sample_next(prev_id, history):
    z = LOGITS[prev_id].astype(float)

    # repetition penalty (CTRL-style): discount logits of seen tokens
    if REP_PENALTY != 1.0:
        for tid in set(history):
            z[tid] = z[tid] / REP_PENALTY if z[tid] > 0 else z[tid] * REP_PENALTY

    z = z / max(1e-6, TEMPERATURE)   # temperature scaling
    p = softmax(z)

    # top-k: keep the k highest-probability tokens
    if TOP_K > 0:
        keep = np.argsort(p)[::-1][:TOP_K]
        mask = np.zeros_like(p, bool); mask[keep] = True
        p = np.where(mask, p, 0.0)

    # min-p: keep tokens with prob >= MIN_P * max(prob) (scales with confidence)
    if MIN_P > 0:
        thresh = MIN_P * p.max()
        p = np.where(p >= thresh, p, 0.0)

    # top-p (nucleus): smallest set whose cumulative prob >= TOP_P
    if 0 < TOP_P < 1:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, TOP_P) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(p, bool); mask[keep] = True
        p = np.where(mask, p, 0.0)

    p = p / p.sum()                                 # renormalise
    return int(np.random.choice(len(VOCAB), p=p))

if __name__ == "__main__":
    cur = 0
    out = [VOCAB[cur]]
    hist = [cur]
    for _ in range(12):
        cur = sample_next(cur, hist)
        hist.append(cur)
        out.append(VOCAB[cur])
    print(" ".join(out))
`;

export const attentionPython = (scale = 1, causal = false, heads = 1, head = 0) => `import numpy as np

# Multi-head self-attention — mirrors the lab.
TOKENS = ["The", "cat", "sat", "on", "the", "mat"]

# tiny fixed embeddings (d = 4); Q = K = V = embeddings (identity projections)
E = np.array([
    [ 1.0,  0.2, -0.5,  0.1],   # The
    [ 0.9,  1.0,  0.2, -0.3],   # cat
    [-0.2,  0.8,  1.0,  0.4],   # sat
    [ 0.1, -0.4,  0.6,  1.0],   # on
    [ 1.0,  0.2, -0.5,  0.1],   # the
    [-0.3,  0.7,  0.9,  0.5],   # mat
])
N, d = E.shape
SCALE = ${scale}        # extra softmax temperature (lab slider)
CAUSAL = ${causal ? 'True' : 'False'}   # causal mask: a token only attends to itself + the past
HEADS = ${heads}        # number of attention heads (d is split across heads)
HEAD = ${head}          # which head this run visualises

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def head_attention(h):
    dh = d // HEADS
    sl = slice(h * dh, (h + 1) * dh)            # this head sees a slice of the dims
    Qh, Kh, Vh = E[:, sl], E[:, sl], E[:, sl]
    scores = Qh @ Kh.T / (np.sqrt(dh) * SCALE)  # N x N relevance scores
    if CAUSAL:
        mask = np.triu(np.ones((N, N), bool), 1) # upper triangle = future
        scores = np.where(mask, -1e9, scores)    # forbid attending to the future
    A = softmax(scores, axis=1)                  # row-wise attention weights
    return A, A @ Vh                             # weights, context-mixed outputs

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    A, out = head_attention(HEAD)
    print(f"head {HEAD}/{HEADS}  causal={CAUSAL}  (rows = query token):")
    print("    " + " ".join(f"{t:>5}" for t in TOKENS))
    for t, row in zip(TOKENS, A):
        print(f"{t:>4} " + " ".join(f"{v:5.2f}" for v in row))
`;
